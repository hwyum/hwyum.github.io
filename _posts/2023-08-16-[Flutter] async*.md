---
title: Flutter의 generator function 만들기
date: 2023-08-16
tags: 
    - flutter
    - async
    - generator
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

* reference: https://medium.flutterdevs.com/dart-generators-callable-class-in-flutter-e4b0b47bd1cf

## Dart Generator

Dart에서의 Generator는 요청이 있을 때마다 일련의 값들을 lazy하게 생성하는 데에 사용되는 함수이다. 이런 값들은 동기적/비동기적으로 생성될 수 있다. 

