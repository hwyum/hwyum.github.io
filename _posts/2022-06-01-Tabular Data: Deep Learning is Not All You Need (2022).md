---
title: "Tabular Data: Deep Learning is Not All You Need (2022)"
date: 2022-06-01
tags:
    - ML 
    - Tabular Data 
    - XGBoost
categories: 
    - ML Papers
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

# 1. Intrudoction



# 2. Background

## Tabular Data에서의 딥러닝

2개의 메인 카테고리가 존재함. 

- Differentiable trees: 
  - DT를 미문 가능하게 만드는 방법을 찾는 것.
  - 기존 DT는 미분할 수 없어 gradient optimization이 불가능. => 내부 노드에서의 decision function을 스무딩해서 tree function과 tree rountingAttention-based models
- Attention-based models:
  - tabular deep network에 attention과 같은 모듈을 도입함. 
  - 최근의 연구들은, 'inter-sample' attention (주어진 샘플 내 피쳐간 interaction)과 'intra-sample attention'(데이터 포인트간의 interaction)을 모두 제안하였음. 
  - 예시:
    - TabNet (Arik and Pfister, 2021)
    - Tabtransformer (Huang, 2020)
- Etc. 
  - Regularization method: 대규모 하이퍼파라미터 튜닝 방식을 사용하여 모든 신경 가중치(neural weight)에 대해 "regularization strength"를 학습 (??!!)
  - Multiplicative interaction에 대한 명시적인 모델링: 여러 가지 방식으로 feature product를 MLP 모델로 통합하고자 함. 
  - 1D-CNN: covolution의 이점을 tabular data에서 활용함. 

그러나, 앞서 언급했듯이 아직 벤치마크 데이터셋이 없어서 모델간 비교가 잘 되고 있지는 않습니다. 



## 비교 대상 모델

본 연구에서는 4개의 모델을 고려했습니다. 

- TabNet (Arik and Pfister, 2021) / Google
- NODE(Popov et al., 2020)
- DNF-Net (Katzir et al., 2021)
- 1D-CNN (Baosenguo, 2021)



### TabNet

(참고: https://lv99.tistory.com/83)

- Encoder가 존재하고, 인코더 안에서 시퀀셜한 결정단계가 sparse한 학습된 마스크를 사용하여 피쳐를 인코딩하고,
- 각 row별로 attention을 활용하여 유의미한 피쳐를 선택하게 됩니다. Sparsemax layer들을 활용해서 인코더는 작은 셋의 피쳐들을 선택하게 됩니다.
  - TabNet Encoder가 Feature Engineering효과를 내고, decision making 부분을 통해 feature selection이 이루어짐.
  - Sequential Attention을 사용하여 각 의사 결정 단계에서 추론할 피쳐를 선택하여 학습 능력이 가장 두드러진 기능에 사용되므로, interpretability와 효과적인 학습을 가능하게 함. 
  - ![img](https://blog.kakaocdn.net/dn/ELrsC/btqNMM0LiTK/L7JEndHQxq8mMmLt6ohVpk/img.png)

- Learning mask의 장점은 피쳐가 all-or-nothing이 아니라는 것입니다. feature에 대한 hard threshold를 사용하기 보다, learnable mask는 soft decision을 할 수 있게 만들어서, 기존의 미분이 불가능한 피쳐 셀렉션 방식을 완화시키게 됩니다. 
- 모델 아키텍쳐
  - ![img](https://blog.kakaocdn.net/dn/bDqL4v/btqNLoffvaX/Sqka1Xhgwz4uEoMJuyVwh1/img.png)
  - 



### 	

### Neural Oblivious Decision Ensembles (NODE)

참고: https://deep-and-shallow.com/2021/02/25/neural-oblivious-decision-ensemblesnode-a-state-of-the-art-deep-learning-algorithm-for-tabular-data/

- NODE 네트워크는 같은 depth의 Oblivious Decision Tree를 포함합니다. ODT들은 미분이 가능하여 error gradient 들이 backpropagate 가능합니다. 

  - ODT도 전통적인 Decision Tree와 마찬가지로 선택된 피쳐에 대해 데이터를 split하고, 각 피쳐를 학습된 threshold와 비교합니다. 
  - 그러나, 각 레벨에서 '단 하나'의 피쳐만 선택되어서, 그 결과로 미분 가능한 균형잡힌 ODT가 됩니다. 
  - 완성된 모형은 미분 가능한 트리들의 앙상블이 됩니다. 

- Decision Tree를 어떻게 미분가능하게 만드는가?

  - 일반적인 DT는 다음과 같습니다. 

    ![img](https://deepandshallowml.files.wordpress.com/2020/12/image-1.png)

    - 여기에서 leaf node는 class label에 대한 분포 이므로, 이를 sigmoid나 softmax activation으로 대체할 수 있을 것입니다. -> 이것은 미분 가능합니다. 

    - Decision node를 살펴봅니다. Decision node의 목적은, 샘플을 left나 right로 라우팅시킬 건지에 대한 의사결정을 합니다. 이러한 decision들을 $d_{i}$와 $\bar{d}_{i}$ 라고 부릅시다. 

      