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



아래의 예시를 통해 확인해봅니다. 

아래는, 누가 회사를 그만둘지를 예측하는 모형이고, employee의 배경과 회사와의 관계와 연관된 피쳐를 가지고 예측하는 문제를 해결하고자 합니다. 

간단한 전처리, missing value에 대한 imputing과 원핫 인코딩을 거친 후, 모델을 학습하고 튜닝해서 96.1%의 정확도를 얻었습니다. 그리고, 테스트 데이터셋 데이터(전체 데이터의 25%)에서 재현율 85%를 달성했습니다. (즉, 진자 퇴사한 사람의 85%를 퇴사할 것이라고 예측한 것)

회사의 CEO가, 어떤 직원이 떠날 것인지 예측함에 있어 어떤 요소가 중요한 영향을 미쳤는지 판단하는 것이 중요할 것입니다. -> 이것이 피쳐 중요도(feature importance)가 하는 중요한 역할입니다. 

![img](https://miro.medium.com/max/1400/1*pLtutNvjMO_7hE0iCmwb4Q.png)

위 그래프를 살펴보면, Monthly Income이 가장 중요한 결정 요소인 것으로 보이고, 이건 상식에도 부합합니다. 



그런데, 간단한 실험을 한번 해봅시다. 
