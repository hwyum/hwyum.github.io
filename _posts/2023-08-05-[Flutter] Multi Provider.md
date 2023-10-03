---
title: "Flutter main에서 여러개의 provider 추가하기"
date: 2023-08-05
tags: 
    - flutter
    - provider
    - 상태관리
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

Flutter main에서 여러 개의 Provider를 추가하고, 상태를 관리하려면 아래와 같이 runApp()의 인자로 `MultiProvider`를 추가해줄 수 있다. 

```dart
void main() async {
  logger.d('app started!!');

  runApp(MultiProvider(providers: [
    ChangeNotifierProvider(create: (context) => UserNotifier()),
    ChangeNotifierProvider(create: (context) => CategoryNotifier()),
  ], child: const MyApp()));
}
```

