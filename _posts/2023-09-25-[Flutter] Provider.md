---
title: Flutter Provider
date: 2023-09-25
tags: 
    - flutter
    - provider
    - stage_management
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

flutter의 상태 관리 방법에 대해서는 자세히 알아 두어야 할 것 같다. 

내가 처음 접한 방식은 `Provider`를 활용하는 것이다. 

`코딩쉐프님`의 강좌에 개념이 잘 설명되어 있어 해당 내용을 기록해둔다.

---

# Reference

- 코딩쉐프님 강의
  - https://www.youtube.com/watch?v=-3iD7f3e_SU&list=PLQt_pzi-LLfoVM7n46d3PY1cloRy_gDpX&index=1
  - https://www.youtube.com/watch?v=de6tAJS2ZG0&list=PLQt_pzi-LLfoVM7n46d3PY1cloRy_gDpX&index=2

# State Management란?

## State란?

- State란 어떤 형태이든 앱 화면 즉, UI에 변화가 생기도록 영향을 미치는 데이터이다. 
- 데이터에는 앱 수준의 데이터와, 위젯 수준의 데이터가 있다. 
  - 앱 수준의 데이터는 서버와 연동해서 또는 서버에 저장된 정보를 끌어와서 앱의 화면에 변화를 일으키는 모든 데이터를 의미한다. 
  - 위젯 수준의 데이터는 사용자가 체크박스를 하면 위젯에서 처리하는 것과 같이 위젯에서 처리하는 수준의 데이터를 의미한다.

## State Management란?

- '효율성'과 관련이 있는 개념
- 기본적으로 플러터는 hot reload를 통해 위젯트리를 리빌드한다. 

### 예시

- 다음 앱은 플러스 버튼을 누르면 플러스 포인트가 올라가고, 현재의 최종 포인트를 보여주면서 격려의 메시지를 출력한다. 마이너스 버튼을 눌러서 최종 포인트가 0보다 작아지면 더 열심히 노력하라는 메시지를 출력한다. 하단에는 최종 스코어를 표시하고 있다. 
  - <img src="/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925180357696.png" alt="image-20230925180357696" style="zoom:15%;" /><img src="/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925180446652.png" alt="image-20230925180446652" style="zoom:15%;" /><img src="/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925180606559.png" alt="image-20230925180606559" style="zoom:15%;" />

- 위젯트리를 살펴보면 다음과 같다. 최상위 위젯에서 생성한 변수를 통해 데이터에 접근하게 된다. 

  - ![image-20230925180829705](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925180829705.png)

   - 여기에서 중요한 것은, 최상위 위젯에서 하위 데이터를 내려보내기 위해서  메서드를 선언하고 있다는 것이다. 
     - 해당 메서드에서 setState()를 호출하게 되면 MyPage 위젯 하에 있는 모든 위젯을 모두 다시 빌드하게 된다. 이것은 매우 비효율적이다. 

### setState method

- 플러터는 state가 변하면 setState method를 활용해서 위젯을 다시 빌드하게 된다. 
- setState method는 플러터가 기본으로 제공하는 아주 심플한 상태 관리 방법이라고 볼 수 있다. 

- 위의 예제로 다시 돌아가보면, setState method가 호출되면 상태값에 따라 변하지 않는 위젯의 경우에도 다시 다 빌드되어야 하는 비효율이 발생하게 된다. 

- 또 하나의 문제점으로 한 위젯에서 setState() 메서드를 사용해서 다시 렌더링을 한 것이 또 다른 위젯에서는 이 변화를 알 수가 없다는 것이다. (아래 그림에서 A 위젯의 변화를 B위젯에서는 알 수 없게 된다. )
  - ![image-20230925185252013](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925185252013.png)

- 정리하자면, setState method 사용시 아래와 같은 문제점이 발생하게 된다. 

  1. 비효율성

  2. 동시에 다른 위젯의 state를 업데이트 시켜주지 못함

### State management 정의

State management를 통해 아래 내용이 가능해져야 한다. 

1. 위젯이 쉽게 데이터에 접근할 수 있도록 한다.
2. 변화된 데이터에 맞춰 UI를 다시 그려준다.

# Provider란?

위에서 설명한 State management를 하기 위한 도구이다. 

아래의 예제를 살펴보자. 위젯 트리가 다음과 같이 정의되어 있다. 

![image-20230925185727367](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925185727367.png)

- 먼저 데이터 모델인 FishModel class를 정의한다. 

- 위의 위젯 트리에 따라 코드를 작성한다.

  - <img src="/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925185900503.png" alt="image-20230925185900503" style="zoom:50%;" />

- 프로바이더 사용 문법은 아래와 같다. 

  - `Provider.of<Salmon>(context) => data`

    - Salmon은 데이터 타입을 의미한다. 
    - Context는 데이터를 흘려주는 수로를 의미한다. 
    - Provider는 데이터를 반환하게 된다. 

    `final fish = Provider.of<Salmon>(context);`

  - 이제 fish 변수는 필요할 때마다 불러서 사용할 수 있게 된다. 

- 플러터에서는 Provider는 하나의 위젯으로 취급되고 일반 위젯과 똑같은 특성을 가지게 된다. 

## Provider 구현

- Provider를 구현하기 위해서는 최상위 위젯을 Provider로 감싸주어야 한다. 
  - ![image-20230925190554196](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925190554196.png)
- Provider에서 요구하는 필수 argument인 create 메서드를 통해 FishModel class를 리턴하도록 한다. 그럼 하위 위젯에서 FishModel 인스턴스에 접근할 수 있게 된다. (수로 개통!!)
  - ![image-20230925191106279](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925191106279.png)
- 하위 위젯에서 아래와 같이 접근할 수 있다. 
  - ![image-20230925191135468](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925191135468.png)
- 결과를 확인하면 아래와 같이 데이터를 가져와 화면에 출력하는 것을 확인할 수 있다. 
  - ![image-20230925191214136](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230925191214136.png)

## 데이터가 변경되었을 때 UI를 다시 그려주는 기능

- 전제는 영향을 받는 위젯만 업데이트 한다는 것이다. 

## ChangeNotifierProvider

