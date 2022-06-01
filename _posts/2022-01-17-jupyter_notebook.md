---
title: "conda 가상환경을 jupyter notebook kernel로 설정하기"
date: 2022-01-17
tags:
    - python
    - jupyter
categories: 
    - python
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---





가상환경을 주피터 노트북을 kernel로 연결시키는 방법에 대한 부분입니다. 



# Reference

- https://taehooh.tistory.com/entry/Python-%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EA%B5%AC%EC%84%B1-%EB%B0%8F-%EC%A3%BC%ED%94%BC%ED%84%B0-%EB%85%B8%ED%8A%B8%EB%B6%81-%EC%BB%A4%EB%84%90-%EC%97%B0%EA%B2%B0



# 커널 연결

- 가상 환경을 활성화 시켜줌

- 가상 환경이 활성화 되어 있는 상황에서 Jupyter notebook 설치

  - ```bash
    pip install jupyter notebook
    ```

    

* 아래 코드를 이용해서 연결한 가상환경이름, 그리고 jupyter notebook에 표시할 이름을 설정할 수 있음. 

```bash
python -m ipykernel install --user --name 가상머신이름 --display-name "표시할이름"
```

