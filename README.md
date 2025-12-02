<p align="center">
  <h1 align="center">✨ <b>DiverseAR</b></h1>
  <h3 align="center">Boosting Diversity in Bitwise Autoregressive Image Generation</h3>
</p>

<p align="center">
  <a href="#">Ying Yang*</a>,
  <a href="#">Zhengyao Lv*</a>,
  <a href="#">Tianlin Pan</a>,
  <a href="#">Haofan Wang</a>,
  <a href="#">Binxin Yang</a>,
  <a href="#">Hubery Yin</a>,
  <a href="#">Chen Li†</a>,
  <a href="#">Chenyang Si†</a>
</p>

<p align="center">
PRLab, Nanjing University · The University of Hong Kong · UCAS · Lovart AI · WeChat, Tencent Inc.<br>
<i>*Equal Contribution  †Corresponding Author</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat&logo=arxiv" height="22">
  </a>
  <a href="https://diverse-ar.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=google-chrome" height="22">
  </a>
</p>

## 🎥 Video Demo

<video width="800" controls>
  <source src="video_demo2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>



## 🔥 The Reason for Bitwise AR Model Degrades Diversity

In bitwise autoregressive models, the predicted probabilities often become **overly peaked**, giving one class **near-certain confidence**.  
This causes **top-p sampling to collapse into a deterministic choice**, removing randomness.  
As shown in the two subplots below, this **collapse of bit-level randomness** restricts feature variation, and diversity degradation mainly arises from  
**(1) the binary classification nature** of bitwise modeling and  
**(2) the overconfident output distributions**.

![Reason Figure](https://github.com/diverse-ar/diverse-ar.github.io/raw/main/diverse5_01.png)

**_※ VAR (Visual Autoregressive Modeling) does not inherently lack diversity; the limitation primarily arises from bitwise tokenization._**

---

## 🚀 The DiverseAR Framework

**DiverseAR** is an effective approach that enhances image and video diversity in **bitwise autoregressive modeling** without sacrificing visual quality.  
It introduces **adaptive logits smoothing** and an **energy-based generation path selection** strategy to achieve richer, more diverse sampling.

![Framework Figure](https://github.com/diverse-ar/diverse-ar.github.io/raw/main/diverse4_01.png)

