<p align="center">
  <h1 align="center">✨ <b>DiverseAR</b></h1>
  <h3 align="center">Boosting Diversity in Bitwise Autoregressive Image Generation</h3>
</p>

<p align="center">
  <h1 align="center"><b>DiverseAR</b></h1>
  <h3 align="center">Boosting Diversity in Bitwise Autoregressive Image Generation</h3>
</p>

<p align="center">
  <a href="#">Ying Yang<sup>1*</sup></a>,
  <a href="#">Zhengyao Lv<sup>2*</sup></a>,
  <a href="#">Tianlin Pan<sup>1,3</sup></a>,
  <a href="#">Haofan Wang<sup>4</sup></a>,
  <a href="#">Binxin Yang<sup>5</sup></a>,
  <a href="#">Hubery Yin<sup>5</sup></a>,
  <a href="#">Chen Li<sup>5†</sup></a>,
  <a href="#">Chenyang Si<sup>1†</sup></a>
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
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
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
- [ ] Code coming soon  



## 🔥 The Reason for Bitwise AR Model Degrades Diversity

In bitwise autoregressive models, the predicted probabilities often become **overly peaked**, giving one class **near-certain confidence**.  
This causes **top-p sampling to collapse into a deterministic choice**, removing randomness.  
As shown in the two subplots below, this **collapse of bit-level randomness** restricts feature variation, and diversity degradation mainly arises from  
**(1) the binary classification nature** of bitwise modeling and  
**(2) the overconfident output distributions**.

![Reason Figure](https://github.com/diverse-ar/diverse-ar.github.io/raw/main/diverse5_01.png)

**_※ VAR (Visual Autoregressive Modeling) does not inherently lack diversity; the limitation primarily arises from bitwise tokenization._**

---

## 🧩 The DiverseAR Framework

**DiverseAR** is an effective approach that enhances image and video diversity in **bitwise autoregressive modeling** without sacrificing visual quality.  
It introduces **adaptive logits smoothing** and an **energy-based generation path selection** strategy to achieve richer, more diverse sampling.

![Framework Figure](https://github.com/diverse-ar/diverse-ar.github.io/raw/main/diverse4_01.png)

