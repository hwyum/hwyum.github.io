---
title: Flutter Completer
date: 2023-11-03
tags: 
    - flutter
    - Completer
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

# Reference

https://api.flutter.dev/flutter/dart-async/Completer-class.html

https://medium.com/@jessicajimantoro/completer-in-dart-flutter-e15f5739e96d

# What is Completer?

- [**Completer**](https://api.flutter.dev/flutter/dart-async/Completer-class.html) is an object that allows you to **generate Future objects** and complete them later with a value or error.
- The Completer **only can be completed once**, otherwise it will throw `Uncaught Error: Bad state: Future already completed error`.

# When to use Completer?

- Based on this [Dart Guide](https://dart.dev/guides/language/effective-dart/usage#avoid-using-completer-directly), you are recommended use `Completer` in 2 kinds of low-level code:

  - **New asynchronous primitive** which means you can use it with [*Timer*](https://api.dart.dev/stable/2.18.0/dart-async/Timer-class.html), [*scheduleMicrotask*](https://api.dart.dev/stable/2.18.0/dart-async/scheduleMicrotask.html), and [*Zone*](https://api.dart.dev/stable/2.18.0/dart-async/Zone-class.html).
    *새로운 비동기 기본 요소 => Timer, scheduleMicrotask, and Zone*

  - **Asynchronous code that doesn’t use futures**

    Future를 사용하지 않는 비동기 코드

  ## Another case to use Completer

  - Callback-based API or Database query
    When their return is not a Future, instead of passing the callback using Future value, you can just use the Completer.
    API, DB query 의 리턴 타입이 Future가 아닌 경우에, 
  - Complex async codes
    **Don’t** use Completer for a simple case, in a way that you can just simply return a `Future`.

# Example

아래와 같이 GoogleMapController Type을 반환하는 Competer를 정의한다. 

```dart
Completer<GoogleMapController> _controller = Completer();
...

@override
Widget build(BuildContext context) {
  return Scaffold(
    appBar: AppBar(
      title: const Text('Simple Google Map'),
      centerTitle: true,
    ),
    body: GoogleMap(
      initialCameraPosition: initalPosition,
      mapType: MapType.normal,
      // Google Map을 실행할 때 받아오는 controller를 conpleter가 완료되었을 때의 값으로 정의해준다.
      onMapCreated: (GoogleMapController controller) {
        _controller.complete(controller);
      },
    ),
  );
}
```



