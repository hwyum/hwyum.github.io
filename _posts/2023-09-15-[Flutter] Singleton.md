---
title: Flutter Singleton Pattern
date: 2023-09-15
tags: 
    - flutter
    - singleton
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

다음의 Reference를 발췌 및 번역하였습니다. 

https://flutterdesk.com/flutter-singleton/



# Dart Singleton Implementation

Dart에서의 singleton 구현은 factory 생성자를 통해 쉽고 유연하게 가능하다. 생서자를 private으로 선언함으로써, classr가 정의된 파일 바깥에서는 초기화가 될 수 없다. 



## Singleton with Factory Constructor

```dart
class SimpleSingleton { 
  static final SimpleSingleton _instance = SimpleSingleton._internal(); 
  factory SimpleSingleton() => _instance; 
  SimpleSingleton._internal(); 
  
  int a = 0; 
  void inc() { 
    a++; 
  } 
} 

main() { 
  final s1 = SimpleSingleton(); 
  final s2 = SimpleSingleton(); 
  print(s1.a); 
  s1.inc(); 
  print(s2.a); 
}
```

## Dart singleton with static getter

```dart
class SimpleSingleton {
  SimpleSingleton._internal();
  static final SimpleSingleton _instance = SimpleSingleton._internal();
  
  static SimpleSingleton get instance => _instance;
  int a = 0;
  void inc() {
    a++;
  }
}
main() {
  final s1 = SimpleSingleton.instance;
  final s2 = SimpleSingleton.instance;
  print(s1.a);
  s1.inc();
  print(s2.a);
}
```

## Singleton with static fields

```dart
class SimpleSingleton {
  SimpleSingleton._internal();
  static final SimpleSingleton instance = SimpleSingleton._internal();
  int a = 0;
  void inc() {
    a++;
  }
}
main() {
  final s1 = SimpleSingleton.instance;
  final s2 = SimpleSingleton.instance;
  print(s1.a);
  s1.inc();
  print(s2.a);
}
```

# Flutter Singleton Example

## FirebaseAUth as a singleton

Firebase plugin들은 대부분 Singleton으로 구현된다. 아래는 FirebaseAuth를 singleton으로 구현한 예제이다. 

```dart
class FirebaseSingleton {
  FirebaseSingleton._private();
  static final _instance = FirebaseSingleton._private();
  factory FirebaseSingleton() => _instance;
  final _auth = FirebaseAuth.instance;
  bool get isLoggedIn => _auth.currentUser != null;
  Future<User> signInWithEmailAndPassword(
    String email,
    String password,
  ) async {
    try {
      final user = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      if (user.user == null) {
        throw 'Login failed';
      }
      return user.user!;
    } on FirebaseException catch (e) {
      switch (e.code) {
        case 'invalid-email':
          throw 'Email is invalid';
        case 'user-not-found':
          throw 'User is not found please sign up';
        case 'wrong-password':
          throw 'Email or password is incorrect';
        default:
          throw 'Unable to sign in please try again later';
      }
    } catch (_) {
      throw 'Unable to sign in please try again later';
    }
  }
  Future<User> signUpWithEmailAndPassword({
    required String email,
    required String password,
  }) async {
    try {
      final createdUser = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
      if (createdUser.user == null) {
        throw 'Login failed';
      }
      return createdUser.user!;
    } on FirebaseException catch (e) {
      if (e.code == 'weak-password') {
        throw 'The password provided is too weak.';
      } else if (e.code == 'email-already-in-use') {
        throw 'The account already exists for that email.';
      }
      throw e.message ?? 'Internet connection error';
    } catch (_) {
      rethrow;
    }
  }
}
```

## Creating a Singleton SharedPreference in Flutter

flutter에서 shared preference는 앱을 종료한 이후에도 저장하기를 원하는 app-level data를 저장하고 다시 받아올 수 있게 해준다. Shared Preference를 Singleton Dart object로 생성하고, 이를 글로벌하게 사용하는 것은 많은 작업들을 단순화 시켜줄 수 있다. 

```dart
class SharedPrefSingleton{
  static final SharedPrefSingleton _instance = SharedPrefSingleton._internal();
  factory SharedPrefSingleton() => _instance;
  SharedPrefSingleton._internal();
  
  late SharedPreferences _pref;
  Future<void> initialize() async {
    _pref = await SharedPreferences.getInstance();
  }
  Future<bool> setName(String name) => _pref.setString('name_key', name);
  String get name => _pref.getString('name_key') ?? '';
}

main() async {
  await SharedPrefSingleton().initialize();
  await SharedPrefSingleton().setName('Flutter');
  print(SharedPrefSingleton().name);
}
```

## Usage of DIO Singleton in Flutter

아래는 http 통신을 위해 사용되는 DIO 패키지를 Singleton으로 활용하는 예제이다.

```dart
class DioSingleton {
  static final DioSingleton _instance = DioSingleton._internal();
  factory DioSingleton() => _instance;
  DioSingleton._internal();
  
  final _client = Dio(BaseOptions(
  	baseUrl: 'https://',
  ));
  Future<void> postData() async {
    try {
      /// POST Request
      await _client.post('path');
    } catch (_) {
      rethrow;
    }
  }
  /// ALL other requests
}
```

## Flutter Provider Singleton

다음은 Provider를 Singleton으로 구현하는 예제이다. 

나도 지금 여기저기에서 Provider 객체를 생성하고 있는데, Singleton으로 수정이 필요할 것 같다. 

```dart
import 'dart:developer';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ProviderSingleton()),
      ],
      child: MaterialApp(
        title: 'SingleTon',
        theme: ThemeData(primarySwatch: Colors.blue),
        home: const MyHomePage(),
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}
class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    final providerSingleton = Provider.of<ProviderSingleton>(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Singleton'),
        actions: [
          IconButton(
            onPressed: () {
              Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const SecondPage(),
              ));
            },
            icon: const Icon(Icons.navigate_next_sharp),
          ),
        ],
      ),
      body: Center(
        child: Text(
          'Count - ${providerSingleton.count.toString()}',
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: providerSingleton.increment,
        child: const Icon(Icons.add),
      ),
    );
  }
}

class ProviderSingleton extends ChangeNotifier {
  ProviderSingleton() {
    log('Creating Singleton of ProviderSingleton');
  }
  var _count = 0;
  int get count => _count;
  void increment() {
    _count++;
    notifyListeners();
  }
}

class SecondPage extends StatelessWidget {
  const SecondPage({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) {
    final providerSingleton = Provider.of<ProviderSingleton>(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Second Page')),
      body: Center(
        child: Text(
          'Count Here - ${providerSingleton.count.toString()}',
        ),
      ),
    );
  }
}
```

