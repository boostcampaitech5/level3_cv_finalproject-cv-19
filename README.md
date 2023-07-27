# level3_cv_finalproject-cv-19
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Introduction
주제 : M. S. G (Mobile Smart Gallary)

목표 : 문장 기반 이미지 검색이 가능한 갤러리 앱 서비스

사용 모델 : CLIP(Contrastive Language-Image Pre-training)

프로젝트 개발 환경 : Ubuntu 18.04.5 LTS, Intel(R)   Xeon(R) Gold 5120 CPU @ 2.20GHz, Ram 90GB, Tesla V100 32GB

---
## Contributors
**`V.I.P`** **`==`** 
**`✨Visionary Innovative People✨`** 
<table>
  <tr>
    <td align="center"><a href="https://github.com/seoin0110"><img src="https://github.com/seoin0110.png" width="100px;" alt=""/><br /><sub><b>김서인</b></sub></a><br /><a href="https://github.com/seoin0110" title="Code"></td>
    <td align="center"><a href="https://github.com/moons98"><img src="https://github.com/moons98.png" width="100px;" alt=""/><br /><sub><b>문상인</b></sub></a><br /><a href="https://github.com/moons98" title="Code"></td>
    <td align="center"><a href="https://github.com/jaehun-park"><img src="https://github.com/jaehun-park.png" width="100px;" alt=""/><br /><sub><b>박재훈</b></sub></a><br /><a href="https://github.com/jaehun-park" title="Code"></td>
    <td align="center"><a href="https://github.com/adam1206"><img src="https://github.com/adam1206.png" width="100px;" alt=""/><br /><sub><b>이강민</b></sub></a><br /><a href="https://github.com/adam1206" title="Code"></td>
     <td align="center"><a href="https://github.com/Jeon-jisu"><img src="https://github.com/Jeon-jisu.png" width="100px;" alt=""/><br /><sub><b>전지수</b></sub></a><br /><a href="https://github.com/Jeon-jisu" title="Code"></td>
  </tr>
</table>

### 역할
|팀원|역할|
|-----|---|
|김서인| 앱 / 백엔드 개발 |
|문상인| 모델링 |
|박재훈| PM, 앱 개발 |
|이강민| 모델링 |
|전지수| 앱 개발 |

## Tech Skill

 <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white"> <img src="https://img.shields.io/badge/React Native-61DAFB?style=for-the-badge&logo=React&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white"> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white">

---

## Repository 구조
- Repository 는 다음과 같은 구조로 구성되어 있습니다. 

```
├── .github
├── clip
├── codebook
├── static
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

----

## Usage

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda create env motis python=3.6
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html ftfy regex
```
