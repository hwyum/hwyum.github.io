---
title: 화면 방향 설정
date: 2023-11-17
tags: 
    - flutter
    - setPreferredOrientations
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

# Reference

https://woongnemonan.tistory.com/entry/%ED%94%8C%EB%9F%AC%ED%84%B0Flutter-%ED%99%94%EB%A9%B4-%ED%9A%8C%EC%A0%84%EC%84%B8%EB%A1%9C%EB%AA%A8%EB%93%9C-%EA%B3%A0%EC%A0%95

# 플러터에서 화면 방향 설정하기

- import `flutter/service.dart`
- `setPreferredorientations`설정

## 세로 모드 고정

```dart 
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  kakao.KakaoSdk.init(nativeAppKey: KAKAOAPP_KEY);
  provider.Provider.debugCheckInvalidValueType = null;
  // 화면 방향 설정 - 세로 모드 고정
  await SystemChrome.setPreferredOrientations(
    [
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ],
  );
...
}
```

