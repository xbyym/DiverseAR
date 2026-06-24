<p align="center">
  <h1 align="center">✨ <b>DiverseAR</b></h1>
  <h3 align="center">Boosting Diversity in Bitwise Autoregressive Image Generation</h3>
</p>


<p align="center">
  <a href="https://github.com/xbyym" target="_blank">Ying Yang<sup>1*</sup></a>,
  <a href="https://scholar.google.com/citations?user=FkkaUgwAAAAJ&hl=en" target="_blank">Zhengyao Lv<sup>2*</sup></a>,
  <a href="https://tianlinn.com/" target="_blank">Tianlin Pan<sup>1,3</sup></a>,
  <a href="https://haofanwang.github.io/" target="_blank">Haofan Wang<sup>4</sup></a>,
  <a href="https://binxinyang.github.io/" target="_blank">Binxin Yang<sup>5</sup></a>,
  <a href="https://openreview.net/profile?id=~Hubery_Yin1" target="_blank">Hubery Yin<sup>5</sup></a>,
  <a href="https://scholar.google.com/citations?user=WDJL3gYAAAAJ&hl=zh-CN" target="_blank">Chen Li<sup>5†</sup></a>,
  <a href="https://chenyangsi.top/" target="_blank">Chenyang Si<sup>1†</sup></a>
</p>


<p align="center">
  <sup>1</sup>PRLab, Nanjing University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>2</sup>The University of Hong Kong &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>3</sup>UCAS &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>4</sup>Lovart AI &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>5</sup>WeChat, Tencent Inc.
</p>

<p align="center">
  <i>* Equal Contribution &nbsp;&nbsp;&nbsp; † Corresponding Author</i>
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2512.02931">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat&logo=arxiv" height="22">
  </a>
  <a href="https://diverse-ar.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=google-chrome" height="22">
  </a>
</p>

## 🎥 Video Demo

<div align="center">

[![Video](https://raw.githubusercontent.com/xbyym/DiverseAR/main/thumbnail.png)](https://github.com/user-attachments/assets/7596d7ee-2399-428a-8160-28e3af4e8b6c)

</div>



## 🚀 Release

- [x] Paper released  
- [x] Demo video released  
- [x] Code released



## 🔥 The Reason for Bitwise AR Model Degrades Diversity

In bitwise autoregressive models, the predicted probabilities often become **overly peaked**, giving one class **near-certain confidence**.  
This causes **top-p sampling to collapse into a deterministic choice**, removing randomness.  
As shown in the two subplots below, this **collapse of bit-level randomness** restricts feature variation, and diversity degradation mainly arises from  
**(1) the binary classification nature** of bitwise modeling and  
**(2) the overconfident output distributions**.

![Reason Figure](https://github.com/diverse-ar/diverse-ar.github.io/raw/main/diverse5_01.png)

**_※ The diversity issue mainly originates from bitwise tokenization, rather than from VAR (Visual Autoregressive Modeling) itself._**

---


## 🛠️ Core Inference Logic

DiverseAR keeps the original autoregressive model architecture unchanged and only modifies the inference-time decoding process.
The released implementation mainly adds two components: **adaptive temperature adjustment** for bitwise logits and **energy-based path search** for selecting better generation trajectories.

### 1. Adaptive Temperature for Bitwise Logits

For bitwise AR models, each token is predicted as a binary classification problem. During inference, the logits can become too confident, e.g., one bit class has probability close to 1.0.
Instead of using a fixed temperature, DiverseAR dynamically finds a temperature `tau` so that the average maximum probability of the bit distribution is close to a target confidence.

```python
def mean_max_prob(logits, tau):
    probs = softmax(logits / tau, dim=-1)
    return probs.max(dim=-1).values.mean()

def find_tau_for_target(logits, target_conf, tau_min=1.0, tau_max=8.0, steps=20):
    # If the current distribution is already not over-confident,
    # keep the original logits.
    if mean_max_prob(logits, tau_min) <= target_conf:
        return tau_min

    lo, hi = tau_min, tau_max
    for _ in range(steps):
        mid = (lo + hi) / 2
        if mean_max_prob(logits, mid) > target_conf:
            lo = mid      # still too sharp, increase temperature
        else:
            hi = mid      # sufficiently smooth
    return hi

def adaptive_temperature(logits, scale_id, target_schedule):
    target_conf = target_schedule.get(scale_id)
    if target_conf is None:
        return logits

    tau = find_tau_for_target(logits, target_conf)
    return logits / tau
```

The target confidence is scheduled by generation scale. Earlier scales control coarse layout and are smoothed more aggressively, while later scales can keep sharper local details:

```python
target_schedule = {
    0: 0.60,  # coarse scale: encourage more diverse global structure
    1: 0.60,
    2: 0.65,
    3: 0.70,
    4: 0.75,
    5: 0.80,  # fine scale: preserve visual quality and details
}
```

After calibration, the normal sampling step is unchanged:

```python
logits = model(prefix, condition, scale_id)
logits = adaptive_temperature(logits, scale_id, target_schedule)
bits = top_p_sample(logits, top_p=top_p)
```

This directly addresses the collapse of bit-level randomness: if the model is too certain, the logits are smoothed until the bit distribution reaches the target confidence; if the logits are already sufficiently uncertain, the original distribution is preserved.

### 2. Energy-based Generation Path Search

Adaptive temperature improves each local sampling decision, while path search improves the global generation trajectory.
DiverseAR samples multiple candidate continuations at each autoregressive scale, scores them with an energy function, and keeps the best paths for the next scale.

```python
def path_energy(logits, sampled_bits, target_conf, lambda_peak=1.0):
    probs = softmax(logits, dim=-1)

    # Negative log-likelihood of the sampled bit path.
    nll = -log_prob(probs, sampled_bits).mean()

    # Penalize paths whose bit distributions remain overly peaked.
    peak = probs.max(dim=-1).values.mean()
    peak_penalty = relu(peak - target_conf)

    return nll + lambda_peak * peak_penalty

def energy_based_search(model, condition, scales, target_schedule, beam_size, num_candidates):
    beams = [GenerationState.empty()]

    for scale_id in scales:
        candidates = []
        target_conf = target_schedule.get(scale_id, 1.0)

        for state in beams:
            logits = model(state.prefix, condition, scale_id)
            logits = adaptive_temperature(logits, scale_id, target_schedule)

            for _ in range(num_candidates):
                sampled_bits = top_p_sample(logits)
                energy = state.energy + path_energy(logits, sampled_bits, target_conf)
                candidates.append(state.extend(sampled_bits, energy))

        beams = sorted(candidates, key=lambda x: x.energy)[:beam_size]

    return beams[0]
```

In practice, the search is lightweight: it does not train a new model or change the tokenizer. It only expands several stochastic AR paths during inference, evaluates them with the energy score above, and selects the path that better balances visual quality and diversity.

## 🧩 The DiverseAR Framework

**DiverseAR** is an effective approach that enhances image and video diversity in **bitwise autoregressive modeling** without sacrificing visual quality.  
It introduces **adaptive logits smoothing** and an **energy-based generation path selection** strategy to achieve richer, more diverse sampling.

![Framework Figure](https://github.com/diverse-ar/diverse-ar.github.io/raw/main/diverse4_01.png)


## 🎬 Visual Results
Please see the project page for full videos and side-by-side comparisons:  
👉 https://diverse-ar.github.io/

## 📚 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{yang2025diversear,
  title={DiverseAR: Boosting Diversity in Bitwise Autoregressive Image Generation},
  author={Yang, Ying and Lv, Zhengyao and Pan, Tianlin and Wang, Haofan and Yang, Binxin and Yin, Hubery and Li, Chen and Si, Chenyang},
  journal={arXiv preprint arXiv:2512.02931},
  year={2025}
}
```

