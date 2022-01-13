 

````yaml
title: "process mining"
date: 2022-01-3130
tags:
    - process_mining python pm4py
categories: 
    - python
````

Reference: https://pm4py.fit.fraunhofer.de/



## Definition of Process Mining

- Obtaining knowledge of, and insights in, processes by means of analyzing the **event data**, generated during the execution of the process.



## What is Process Mining? - Event Data

예를 들어 아래와 같은 transaction들이 존재한다고 가정해보자.

![image-20220113112528047](/Users/kakao/Library/Application Support/typora-user-images/image-20220113112528047.png)

Event vs Activity

* Activities: within specific order



## What is Process Mining? - Overview

![image-20220113112959487](/Users/kakao/Library/Application Support/typora-user-images/image-20220113112959487.png)

### Process Discovery

![image-20220113113029687](/Users/kakao/Library/Application Support/typora-user-images/image-20220113113029687.png)

* Event Log => Model that represents how that event data looks

![image-20220113113254924](/Users/kakao/Library/Application Support/typora-user-images/image-20220113113254924.png)

* Automated or semi-automated argorithm.

![image-20220113113344049](/Users/kakao/Library/Application Support/typora-user-images/image-20220113113344049.png)

* 어떤 순서로 activity가 진행되는지를 보여주고 있음. 

![image-20220113113604984](/Users/kakao/Library/Application Support/typora-user-images/image-20220113113604984.png)

* 동시에 일어나는 일련의 activity들이 존재함. (Concurrency)
  * 고객 사이드에서 invoice를 청구하는 동시에,
  * 재고 확인하고 패키징 할 수가 있음. 



### Conformance Checking

![image-20220113113843586](/Users/kakao/Library/Application Support/typora-user-images/image-20220113113843586.png)

![image-20220113113959667](/Users/kakao/Library/Application Support/typora-user-images/image-20220113113959667.png)

 ![image-20220113114111084](/Users/kakao/Library/Application Support/typora-user-images/image-20220113114111084.png)



### Process Enhancement

* Enhence the value of the model

![image-20220113114307179](/Users/kakao/Library/Application Support/typora-user-images/image-20220113114307179.png)

![image-20220113114319249](/Users/kakao/Library/Application Support/typora-user-images/image-20220113114319249.png)