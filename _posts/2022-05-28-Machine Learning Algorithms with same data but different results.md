---
title: "Machine Learning Algorithms with same data but different results"
date: 2022-05-28
tags:
    - ML Reproducibility
categories: 
    - ML
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

본 포스팅은 https://machinelearningmastery.com/different-results-each-time-in-machine-learning/ 의 글을 번역/이해한 내용입니다. 

# 1. Help, I’m Getting Different Results!?

(Algorithm과 Model은 어떻게 다른가?)
* Algorithm: Procedure run on data that results in model (e.g. training or learning)
* Model:Data structure and coefficients used to make predictions on data
 
 모델을 학습시키는 과정은 “function approximation”이라고 볼 수 있습니다. 즉, input이 주어졌을 때 output을 만들어 낼 수 있는 함수를 근사하는 것입니다. 
 이런 맥락에서 Supervised Learning은 다음과 같이 정의 됩니다. 
*  Supervised Learning: Automatically learning a mapping function from examples of inputs to examples of outputs. 


# 2. Differences Caused by Training Data

# 3. Differences Caused by Learning Algorithm

# 4. Differences Caused by Evaluation Procedure

# 5. Differences Caused by Platform