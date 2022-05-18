---
title: "Random Forest Feature Importance"
date: 2022-05-18
tags:
    - ML Python RandomForest
categories: 
    - ML
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

아래는 다음의 원글을 번역/이해한 내용입니다. 

https://medium.com/@ali.soleymani.co/stop-using-random-forest-feature-importances-take-this-intuitive-approach-instead-4335205b933f



sklearn에서 가장 일반적인 피쳐 중요도를 계산하는 방법은 Random Forest의 'mean decrease in impurity'를 기반으로 피쳐의 중요도를 계산하는 것입니다. 

그런데, 여기에서 크리티컬한 부분이 존재합니다.

**바로 Tree-based model이 continuous numerical 이나 high cardinality categorical 피쳐들을 과대평가하는 경향이 있다는 것입니다.** 



