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
  - DT를 미분 가능하게 만드는 방법을 찾는 것.
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



### Neural Oblivious Decision Ensembles (NODE)

참고: https://deep-and-shallow.com/2021/02/25/neural-oblivious-decision-ensemblesnode-a-state-of-the-art-deep-learning-algorithm-for-tabular-data/

- NODE 네트워크는 같은 depth의 Oblivious Decision Tree를 포함합니다. ODT들은 미분이 가능하여 error gradient 들이 backpropagate 가능합니다. 

  - ODT도 전통적인 Decision Tree와 마찬가지로 선택된 피쳐에 대해 데이터를 split하고, 각 피쳐를 학습된 threshold와 비교합니다. 
  - 그러나, 각 레벨에서 '단 하나'의 피쳐만 선택되어서, 그 결과로 미분 가능한 균형잡힌 ODT가 됩니다. 
  - 완성된 모형은 미분 가능한 트리들의 앙상블이 됩니다. 

- Decision Tree를 어떻게 미분가능하게 만들 수 있을까요?

  - 일반적인 DT는 다음과 같습니다. 

    ![img](https://deepandshallowml.files.wordpress.com/2020/12/image-1.png)

    - 여기에서 leaf node는 class label에 대한 분포 이므로, 이를 sigmoid나 softmax activation으로 대체할 수 있을 것입니다. -> 이것은 미분 가능합니다. 
    - Decision node를 살펴봅니다. Decision node의 목적은, 샘플을 left나 right로 라우팅시킬 건지에 대한 의사결정을 합니다. 이러한 decision들을 $d_{i}$와 $\bar{d}_{i}$ 라고 부릅시다. 이와 같은 의사결정을 위해, 특정 피쳐 (f)와 threshold(b)를 활용합니다. 이것이 노드의 파라미터 입니다. 
    - 전통적인 DT에서는 이러한 의사결정이 binary decision 입니다; right or left, 0 or 1인 것이죠.   -> 이것도 마찬가지로 Simoid function으로 대체할 수 있지 않을까요? 
    - 이것이 Kontschider et al (Deep Neural Decision Forests, 2015) 에서 제안한 방법입니다. 
      - 만약 우리가 0-1의 엄밀한 의사결정을 sigmoid 함수를 가지고 확률적인 의사결정으로 변형한다면, 노드는 미분 가능하게 될 것입니다. 

  - 하나의 노드(decision or a leaft node)에서 작동하는 것을 살펴보았습니다. 이것을 확장해봅시다. 위의 그림에서 red path가 decision tree에서 하나의 path가 될 수 있습니다. 

    - 결정론적인 버전에서는, 특정 샘플이 해당 path로 가는지 못가는지만 결정이 됩니다. 
- 만약에 같은 프로세스를 확률적인 용어로 생각하게 된다면, 특정 샘플에서 해당 path로 갈 확률은 해당 패스에 들어가 있는 모든 노드에 있어서 1이 되어야 합니다. 해당 path의 마지막 leaf노드에 도달하기 위해서 말이죠. 
    - 확률론적인 패러다임에서 우리는 전체 path를 따라 샘플이 왼쪽이나 오른쪽으로가는 확률을 곱해서 leaf노드에 도달할 확률을 구할 수 있게 됩니다. 
  - 즉, 샘플이 표시된 leaf node에 도달하는 확률은, $d_1 * \bar{d}_2 * \bar{d}_5$가 됩니다. 
      




#### Neural Oblivious Trees

Oblivious Trees는 symmetric하게 자라는 Tree 입니다. 각 레벨에서 인스턴스들을 왼쪽이나 오른쪽으로 split할 때 같은 피쳐들이 책임을 가지게 됩니다. 대표적인 gradient boosting 방법인 CatBoost에서 oblivious tree들을 활용합니다. 

Oblivisou Tree들은 흥미로운 것이, Decision Table을 $2^d$ (d = depth of the tree) 개 셀로 줄일 수 있습니다. 이것이 모든 것을 단순하게 만들 수 있습니다. 

각각의 ODT들은, $2^d$개 중 하나의 아웃풋을 내게 됩니다. 이것은 d feature-threshold combination을 활용하게 되고, 이것이 ODT의 파라미터들입니다.

수학적으로 ODT는 아래와 같이 정의됩니다. 

 $h(x)=R\left[\mathbb{I}\left(f_{1}(x)-b_{1}\right), \ldots, \mathbb{I}\left(f_{d}(x)-b_{d}\right)\right]{\space} where{\space} \mathbb{I}{\space} = step {\space} function$

tree output을 미분 가능하게 하려면, splitting feature choice(f)와 threshold b를 활용한 비교연산을 continuous하게 바꾸어야 합니다. 

전통적인 Tree에서는, split을 하기 위한 feature choice가 결정적(deterministic)인 의사결정에 따릅니다. 그러나 미분가능성을 위해서는 조금 더 부드러운 접근이 필요합니다. 즉, `weighted sum of features`가 되어야 하며, 각각의 weight는 학습이 됩니다. 일반적으로 피쳐들에 대한 Softmax choice를 생각할 수 있습니다만, 우리는 보다 sparse한 feature selection을 원합니다. 그러한 효과를 위해서 NODE는 학습가능한 피쳐 셀렉션 matrix $F\in R^{d \times n}$에 $\alpha$-entmax transformation (Peters et al., 2019)(??)을 사용합니다. 

	- ![img](https://deepandshallowml.files.wordpress.com/2020/12/image-2.png)

유사하게, Heaviside function (step function)도 two-class entmax로 완화(relax)시킬 수 있습니다. 다른 피쳐들이, 다른 스케일을 가질 수 있도록, entmax를 파라미터 $\tau$로 스케일 합니다. 

$c_i(x)=\sigma_{\alpha}(\frac{f_ix-b_i}{\tau_i})$, where $b_i$ and $\tau_i$ are learnable parameters for thresholds and scales respectively.

Tree는 $c_i$에 의해 두 개의 사이드를 가지게 되므로, $c_i$, $(1-c_i)$를 쌓아서, “choice” tensor $C$를 정의할 수 있다. 
as outer product of all the trees:

$C(x)=\left[\begin{array}{c}\left.c_{1}(x)\right) \\ 1-c_{1}(x)\end{array}\right] \otimes\left[\begin{array}{c}\left.c_{2}(x)\right) \\ 1-c_{2}(x)\end{array}\right] \otimes \ldots \otimes\left[\begin{array}{c}\left.c_{d}(x)\right) \\ 1-c_{d}(x)\end{array}\right]$

This gives us the choice weights, or intuitively the probabilities of each of the $2^{d}$ outputs, which is in the Response tensor. So now it reduced into a weighted sum of Response tensor, weighted by the Choice tensor.
$$
h(x)=\sum_{i_{i}, \ldots i_{d} \epsilon / 0,1{ }^{d}} R_{i_{i}, \ldots i_{d}} \cdot C i_{i}, \ldots i_{d}(x)
$$
The entire setup looks like the below diagram:

![img](https://deepandshallowml.files.wordpress.com/2020/12/image-3.png)

From the Paper [3] 



#### Neural Oblivious Decision Ensembles

The jump from an individual tree to a "forest" is pretty simple. If we have $m$ trees in the ensemble, the final output is the concatenation of $\mathrm{m}$ individual trees $\left[\hat{h_{1}}(x), \ldots, \hat{h_{m}}(x)\right]$

Going Deeper with NODE
In addition to developing the core module(NODE layer), they also propose a deep version, where we stack multiple NODE layers on top of each other, but with residual connections. The input features and the outputs of all previous layers are concatenated and fed into the next NODE Layer and so on. And finally, the final output from all the layers are averaged(similar to the RandomForest).
<img src="https://deepandshallowml.files.wordpress.com/2020/12/image-4.png?w=1024" alt="img" style="zoom:80%;" />
From the Paper [3]



### DNF-Net

DNF-Net의 아이디어는 disjunctive normal formulas (DNF)를 닙 뉴럴네트워크에서 시뮬레이션 하는 것입니다. 

저자들은 Boolean formula들을 부드러운, 미분가능한 버전으로 대체하는 방법을 제안합니다. 

DNF-Net의 핵심 피쳐는 DNNF(disjunctive normal neural form) 블록인데, 해당 블록은 (1) FC layer, (2) DNNF layer (formed by a sofit version of binary conjunctions over literal)로 구성이 됩니다. 완성된 모델은 이러한 DNNF들의 앙상블이 됩니다. 



### 1-D CNN

참고: https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/2nd%20Place%20Solution.pdf

최근에, 캐글에서 Tabular data에 대해 1-D CNN 모형이 가장 좋은 단일 모델 성능을 보인 바 있습니다 ([Baosenguo, 2021](https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution)). 이 모델은 CNN구조가 ‘feature extraction’에 좋은 성능을 보인다는 아이디어에 기반한 것이었습니다. 여전히 tabular data에서 CNN은 거의 활용되지 않습니다. 왜냐하면 피쳐들의 순서가 locality특성을 가지지 않기 때문입니다. 

이 모델에서는, FC Layer가 locality characteristic을 가지는 큰 사이즈의 피쳐를 만드는 데에 사용이 됩니다. 그리고, 여러 개의 1D-Conv layer들이 shorcut-like connection과 함께 따라오게 됩니다. 



## 2.2 Model Ensemble

앙상블 학습은 성능을 향상시키고 분산을 줄이는 방법론으로 잘 알려져있습니다. 여러 개의 모델을 학습시키고, 그들의 결과를 결합하는 것입니다. 

앙상블 학습은, 다른 머신러닝 방법론들이 각기 다른 상황에서 더 성능이 좋거나, 실수를 한다는 가정을 전제하고 있습니다. 

앙상블 학습은 대개 두 개의 메인 타입으로 분류가 됩니다. 

1. Randomization에 기반한 테크닉
   - Random Forest를 그 예로 들 수 있습니다. - 각각의 앙상블 멤버들은 서로 다른 초기 파라미터 설정과 학습 데이터를 가지고 있습니다. 
   - 베이스 러너들은 서로간의 상호작용없이 동시 다발적으로 학습이 가능합니다. 
2. Boosting-based 방법론
   - 베이스 러너들이 연속적으로 (Sequentially) 학습되면서 모델을 적합 시키게 됩니다. 

대부분의 앙상블에서 Decision Tree가 베이스 러너로 활용이 됩니다. 

본 연구에서는 앙상블에 5개 분류기를 포함시켰음: TabNet, NODE, DNF-Net, 1D-CNN, XGBoost

실용적이고 직관적인 앙상블을 구축하기 위해, 두 가지 다른 버전을 제안하였음

	- (1) 각 분류기를 동일한 weight로 취급
	- (2) 각 분류기에서 나온 예측들에 대해 가중 평균을 취함. 

데이터가 많을 수록 일반적으로 성능이 좋기 때문에, 각 모델을 학습 시킬 때에도 전체 학습 데이터를 활용하였음.

# 3. Comparing the Models

## 3.1 Experimental Setup

## 3.2 Results



# 4. Discussion and Conclusions

