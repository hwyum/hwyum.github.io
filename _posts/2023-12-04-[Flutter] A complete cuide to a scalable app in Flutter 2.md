---
title: A complete guide to a scalable app in Flutter#2 Data layer (Medium Article)
date: 2023-12-4
tags: 
    - flutter
    - Clean_Architecture
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

[다음](https://gbaccetta.medium.com/a-complete-guide-to-a-scalable-app-in-flutter-part-2-data-layer-7629b6bb3835) Medium Article 내용을 정리한 글입니다. 

# Implementing of Data layer

Goal: to construct our first model, create the corresponding interactor, and develop a module for fetching data from the Medium RSS feed (tuilizing and API service).

## Our first model: Article

This Article class will encompass essential attributes derived from the Medium RSS feed, including fields for the title, description, content, keywords, URL, date, and an optional cover image.  

```dart
class Article {
  final String title;
  final String description;
  final String content;
  final List<String> keywords;
  final String url;
  final DateTime date;
  final String? coverImage;

  Article({
    required this.title,
    required this.description,
    required this.content,
    required this.keywords,
    required this.url,
    required this.date,
    this.coverImage,
  });
```

## ArticleInteractor

ArticleInteractor, responsible for managing the business logic associated with Article data. This interactor will act as an intermediary, facilitating communication between the data layer and the presentation layer. 

To fetch data from the Medium platform, **we will also need to establish an ApiService, designed to handle HTTP requests** directed towards the Medium API.

### ApiService

For organizational purposes, we’ll establish an `ApiService` class within a dedicated “services” folder

Notably, we’ve made the decision to expose this getter specifically to **facilitate the mocking of the HTTP adapter during unit tests**, a task made convenient by the [http_mock_adapter](https://pub.dev/packages/http_mock_adapter) package.

```dart
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_gbaccetta_feed_app/domain/failures/api_error.dart';
import 'package:flutter_gbaccetta_feed_app/data/modules/api/endpoints.dart';

class ApiService {
  ///this is public will be used to mock [apiService] requests during testing
  final Dio dio = Dio(); // Dio 인스턴스 생성

  bool get _runningTests =>
      kIsWeb == false && Platform.environment.containsKey('FLUTTER_TEST'); // 현재 플랫폼이 웹이 아니고, 환경 변수에 'FLUTTER TEST'가 설정되어 있는지 여부를 반환. -- 테스트 실행 여부 확인에 사용된다. 

  // ApiService의 생성자. 생성자는 객체가 생성 될 때 호출된다. 
  ApiService() {
    if (!kReleaseMode && !_runningTests) { // 앱이 릴리스 모드가 아니고, 테스트 실행 중이 아닐 경우에만 내부 코드를 실행한다. 
      // this line can be excluded from coverage (no interceptor during tests)
      dio.interceptors.add(LogInterceptor(responseBody: true)); // coverage:ignore-line. Dio 인터셉터에 로그 인터셉터를 추가한다. 이를 통해 HTTP 응답의 본문을 로깅할 수 있다. 
    }
  }

  Future<Response> _sendRequest(Future<Response> request) async {
    try {
      return await request;
    } catch (e) {
      throw ApiError(message: e.toString());
    }
  }

  Future<Response> getMediumRssFeed() async {
    return _sendRequest(
      dio.get(Endpoints.mediumRssFeed),
    );
  }
}
```

* LogInterceptor
  * useful when running the app in debug mode. 
* `ApiError` class 정의
* `Endpoints`class 정의

### The ArticleInteractor

* facilitating the interaction between the data layer and the presentation layer. 

* `ArticleInteractor`will adhere to a set of define `article UseCases` encapsulated within the domain layer.

```dart
import 'package:flutter_gbaccetta_feed_app/domain/models/article.dart';

abstract class ArticleUseCases {
  Future<List<Article>> getArticles();
}
```



### The helper factory mothod for our model

```dart
factory Article.fromRssItem(RssItem item) {
  return Article(
    title: item.title ?? '',
    description: item.description ?? '',
    
  )
}
```

