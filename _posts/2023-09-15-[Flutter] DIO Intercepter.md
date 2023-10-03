---
title: DIO Interceptor를 활용해 특정 에러 코드에 대해 처리하기
date: 2023-09-15
tags: 
    - flutter
    - async
    - dio
    - Intercepter
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

# 문제 상황

App을 실행시킨지 오래 되었을 때, Token이 만료가 되어 서버와의 통신이 불가능해 진다. 

이때 내가 개발중인 앱의 서버에서는 403에러를 내려주고 있는데, 이렇게 특정 에러 코드를 만났을 때 다시 login 화면을 보여주는 조치가 필요하다. 

GPT에게 문의해보니 아래와 같이 답변을 해준다. 

참고로 나는 서버와의 통신을 위해 Dio 패티지를 사용하고 있다. 

## GPT에 질의

flutter에서 서버 api 호출시 에러가 떴을 때, 에러 코드에 따라 처리해주려면 어떻게 해? 지금은 try catch로만 되어 있어서 코드에 따라 처리는 못해. 현재 get request를 처리하는 코드는 아래와 같아. (dio를 wrapping하고 있어) 

(... 이하 코드 생략 ...)

## GPT 답변

> Flutter에서 서버 API 호출 중에 발생하는 에러를 에러 코드에 따라 처리하려면 다음과 같이 할 수 있습니다. 에러 처리를 위해 Dio의 `Interceptor`를 사용하여 특정 에러 코드를 감지하고 처리할 수 있습니다. 아래는 이러한 접근 방식의 예제입니다.
>
> 먼저, Dio의 `Interceptor` 클래스를 확장한 커스텀 인터셉터를 작성합니다.
>
> ```dart
> import 'package:dio/dio.dart';
> 
> class ErrorHandlingInterceptor extends Interceptor {
>   @override
>   Future onError(DioError err, ErrorInterceptorHandler handler) async {
>     // 에러 상태 코드를 확인하여 처리
>     if (err.response?.statusCode == 401) {
>       // 예를 들어, 401 Unauthorized 에러를 처리하는 코드
>       // 여기에 처리할 내용을 추가하세요.
>     } else if (err.response?.statusCode == 404) {
>       // 404 Not Found 에러를 처리하는 코드
>       // 여기에 처리할 내용을 추가하세요.
>     } else {
>       // 다른 에러 코드에 대한 기본 처리
>       // 여기에 기본 에러 처리 코드를 추가하세요.
>     }
>     return super.onError(err, handler);
>   }
> }
> 그런 다음, Dio 클라이언트를 만들 때 이 커스텀 인터셉터를 등록합니다.
> ```

```dart
import 'package:dio/dio.dart';

Dio dio = Dio();

void main() {
  // ErrorHandlingInterceptor를 Dio에 추가
  dio.interceptors.add(ErrorHandlingInterceptor());

  // 앱 실행 등 나머지 설정
}
```

> 이렇게 하면 각각의 상태 코드에 대한 에러 처리 로직을 추가할 수 있습니다. onError 메서드에서 에러를 감지하고 처리하는 부분에 필요한 코드를 추가하세요. 위의 예제에서는 401과 404 에러 코드를 처리하는 방법을 보여주었지만, 다른 에러 코드에 대한 처리도 비슷한 방식으로 추가할 수 있습니다.
>
> 그런 다음, getRequest 함수를 호출할 때 발생하는 에러는 이러한 인터셉터를 통해 처리될 것입니다.



사실, 에러 코드에 따른 화면 전환이 필요한데 그렇게 하려면 아래와 같이 class 객체를 생성할 때 context를 받아와서 initialize 해주어야 한다. 

> ```dart
> import 'package:dio/dio.dart';
> import 'package:flutter/material.dart';
> 
> class ErrorHandlingInterceptor extends Interceptor {
>   final BuildContext context;
> 
>   ErrorHandlingInterceptor(this.context);
> 
>   @override
>   Future onError(DioError err, ErrorInterceptorHandler handler) async {
>     // 에러 상태 코드를 확인하여 처리
>     if (err.response?.statusCode == 401) {
>       // 401 Unauthorized 에러를 처리하는 코드
>       // 로그인 화면으로 이동
>       Navigator.of(context).pushReplacementNamed('/login'); // 로그인 화면으로 이동
>     } else if (err.response?.statusCode == 404) {
>       // 404 Not Found 에러를 처리하는 코드
>       // 다른 처리를 추가하거나 다른 화면으로 이동할 수 있습니다.
>     } else {
>       // 다른 에러 코드에 대한 기본 처리
>       // 여기에 기본 에러 처리 코드를 추가하세요.
>     }
>     return super.onError(err, handler);
>   }
> }
> ```
>
> 위의 코드에서 `context`를 전달하고, 401 Unauthorized 에러가 발생했을 때 `/login` 라우트로 화면을 전환하도록 설정되어 있습니다.
>
> 그런 다음, Dio 클라이언트를 만들 때 이 커스텀 인터셉터를 등록합니다.
>
> ```dart
> import 'package:dio/dio.dart';
> 
> Dio dio = Dio();
> 
> void main() {
>   // ErrorHandlingInterceptor를 Dio에 추가
>   dio.interceptors.add(ErrorHandlingInterceptor(context));
> 
>   // 앱 실행 등 나머지 설정
> }
> ```