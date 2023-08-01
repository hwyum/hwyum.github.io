---
title: "[Flutter] Go Router"
date: 2023-07-31
tags:
    - flutter
categories: 
    - flutter
---

# go_router 사용 관련 정리

**아래의 Medium Article 내용을 번역/정리한 것임을 미리 밝힙니다.** 

https://medium.com/@antonio.tioypedro1234/flutter-go-router-the-essential-guide-349ef39ec5b3



## Route Configuration

Route 속성을 정의한다. 

```dart
import 'package:go_router/go_router.dart';

// GoRouter configuration
final _router = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(
      name: 'home', // Optional, add name to your routes. Allows you navigate by name instead of path
      path: '/',
      builder: (context, state) => HomeScreen(),
    ),
    GoRoute(
      name: 'page2',
      path: '/page2',
      builder: (context, state) => Page2Screen(),
    ),
  ],
);
```

 `MaterialApp.router`이나 `CupertinoApp.router` 생성자를 사용해서 앱을 빌드할 때 위에서 정의한 _router를 `routerConfig`라는 parameter로 넘겨주게 된다. 

```dart
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      routerConfig: _router,
    );
  }
}
```

### Parameters

아래와 같이 path parameter를 `:`캐릭터를 사용해 정의할 수 있다. 

```dart
GoRoute(
  path: '/fruits/:id',
  builder: (context, state) {
     final id = state.params['id'] // Get "id" param from URL
     return FruitsPage(id: id);
  },
),
```

`GoRouterState`를 활용해 Query string parameter에도 access 할 수 있다. 예를 들어서 `/fruits?search=antonio`는 `search`파라메터를 읽을 수 있다. 

```dart
GoRoute(
  path: '/fruits',
  builder: (context, state) {
    final search = state.queryParams['search'];
    return FruitsPage(search: search);
  },
),
```

### Adding child routes

`push()`와 마찬가지로, 원 화면이 있고, 그 위에 새로운 화면을 띄울 수 있게 (back button 사용 가능하도록) 한개 이상의 스크린으로도 라우팅 시킬 수 있다. 

이를 위해 child route와 parent route를 설정할 수 있다. 

```dart
GoRoute(
  path: '/fruits',
  builder: (context, state){
    return FruitsPage();
  },
  routes: <RouteBase>[ // Add child routes
    GoRoute(
      path: 'fruits-details', // Note: Don't need to specify "/" character for router's parents
      builder: (context, state) {
        return FruitDetailsPage();
      }
    )
  ]
)
```

### Navigation Between Screens

Go_router를 활용해 navigate하는 데에는 여러 방법이 존재한다. 

새로운 스크린으로 전환하려면,  `context.go()`에 URL을 주면 된다. 

```dart
build(BuildContext context) {
  return TextButton(
    onPressed: () => context.go('/fruits/fruit-detail'),    
  );
}
```

URL대신 이름으로 화면 전환도 가능하다. 

```dart
build(BuildContext context) {
  return TextButton(
    // remember to add "name" to your routes
    onPressed: () => context.goNamed('fruit-detail'),
  );
}
```

## Nested Tab navigation

탭별로 분기된 navigation을 정의할 수도 있다. (bottom navigator를 사용하는 경우, 각 탭별로 navigation tree를 유지해야 하는 경우가 있을 수 있다. )

이러한 nested navigation (<- 왼쪽 단어가 잘 와닿지는 않는다.)을 구현하기 위해서 [`StatefulShellRoute`](https://pub.dev/documentation/go_router/latest/go_router/StatefulShellRoute-class.html)를 활용할 수 있다. 

[`StatefulShellRoute`](https://pub.dev/documentation/go_router/latest/go_router/StatefulShellRoute-class.html)는 `StatefulShellBranch`의 List를 정의함으로써 생성된다. `StatefulShellBranch`는 roote routes와 각 브랜치별 Navigator key(GlobalKey)와 선택적인 initial location(?)을 제공한다. 

router를 생성할 때, StatefulShellRoute.indexedStack() 생성자를 사용하게 된다. 

```dart
// Create keys for `root` & `section` navigator avoiding unnecessary rebuilds
final _rootNavigatorKey = GlobalKey<NavigatorState>();
final _sectionNavigatorKey = GlobalKey<NavigatorState>();


final router = GoRouter(
  navigatorKey: _rootNavigatorKey,
  initialLocation: '/feed',
  routes: <RouteBase>[
    StatefulShellRoute.indexedStack(
      builder: (context, state, navigationShell) {
        // Return the widget that implements the custom shell (e.g a BottomNavigationBar).
        // The [StatefulNavigationShell] is passed to be able to navigate to other branches in a stateful way.
        return ScaffoldWithNavbar(navigationShell);
      },
      branches: [
        // The route branch for the 1º Tab
        StatefulShellBranch(
          navigatorKey: _sectionNavigatorKey,
          // Add this branch routes
          // each routes with its sub routes if available e.g feed/uuid/details
          routes: <RouteBase>[
            GoRoute(
              path: '/feed',
              builder: (context, state) => const FeedPage(),
              routes: <RouteBase>[
                GoRoute(
                  path: 'detail',
                  builder: (context, state) => const FeedDetailsPage(),
                )
              ],
            ),
          ],
        ),

        // The route branch for 2º Tab
        StatefulShellBranch(routes: <RouteBase>[
          // Add this branch routes
          // each routes with its sub routes if available e.g shope/uuid/details
          GoRoute(
            path: '/shope',
            builder: (context, state) => const ShopePage(),
          ),
        ])
      ],
    ),
  ],
);

```

## Guards

특정 route를 보호하려면 (예. 인증받지 않은 유저들에 대해) global `redirect`를 설정할 수 있다. 가장 일반적인 예시로, 유저가 로그인되지 않은 상태일 때 `/login`으로 redirect시키는 것을 생각할 수 있다. 

`redirect`는 `GoRouterRedirect`type의 콜백이다. 

```dart
GoRouter(
  redirect: (BuildContext context, GoRouterState state) {
    final isAuthenticated = // your logic to check if user is authenticated
    if (!isAuthenticated) {
      return '/login';
    } else {
      return null; // return "null" to display the intended route without redirecting
     }
   },
  ...
```

`GoRouter`생성자에 redirect를 정의할 수 있다. 다른 navigation event 이전에 호출된다. 

## Error handling (404 page)

디폴트 설정으로 `MaterialApp`, `CupertinoApp` 모두에 대해 default error screen을 제공한다. `errorBuilder` 파라메터를 통해 default error screen을 변경할 수 있다. 

```dart
GoRouter(
  /* ... */
  errorBuilder: (context, state) => ErrorPage(state.error),
);
```

## Type-safe routes

💡‼️아직 이부분은 잘 이해를 못해서 모든 내용을 담지는 못했다. 

URL string(`context.go("/auth")`)을 사용하는 대신에, `go_router_builder`패키지를 사용해 type-safe routes(?)도 지원한다. 

이를 위해서는 go_router_builder, build_runner, build_verify를 dependency에 추가해야 한다. 

```yaml
dev_dependencies:
  go_router_builder: ^1.0.16
  build_runner: ^2.3.3
  build_verify: ^3.1.0
```

### Defining a route

`GoRouteData`를 상속받고, `build` method를 오버라이드해서 각 route를 정의한다.

```dart
class HomeRoute extends GoRouteData {
  const HomeRoute();
  
  @override
  Widget build(BuildContext context, GoRouterState state) => const HomeScreen()
}
```

### Route tree

라우트 트리(route tree)가 각 top-level 루트의 속성으로 정의된다. 

```dart
import 'package:go_router/go_router.dart';

part 'go_router.g.dart'; // name of generated file

// Define how your route tree (path and sub-routes)
@TypedGoRoute<HomeScreenRoute>(
    path: '/home',
    routes: [ // Add sub-routes
      TypedGoRoute<SongRoute>(
        path: 'song/:id',
      )
    ]
)

// Create your route screen that extends "GoRouteData" and @override "build"
// method that return the screen for this route
@immutable
class HomeScreenRoute extends GoRouteData {
  @override
  Widget build(BuildContext context) {
    return const HomeScreen();
  }
}

@immutable
class SongRoute extends GoRouteData {
  final int id;
  const SongRoute({required this.id}); // Constructor (생성자)

  @override
  Widget build(BuildContext context) {
    return SongScreen(songId: id.toString());
  }
}  
```

생성된 파일을 빌드하려면 `buiuld_runner` 코맨드를 사용한다. 

```dart
flutter pub global activate build_runner // Optional, if you already have build_runner activated so you can skip this step
flutter pub run build_runner build
```

화면을 전환하려면, GoRouterData object를 필요한 파라메터와 함께 생성하고 go()를 콜하면 된다.

```dart
TextButton(
  onPressed: () {
    const SongRoute(id: 2).go(context);
  },
  child: const Text('Go to song 2'),
),
```

## NavigatorObserver

GoRouter에서 Navigator를 관찰하기 위해 (push, pop 또는 대체가 일어날 때 listen) `NavigatorObserver`를 추가할 수 있다. 이를 위해서 `NavigatorObserver`를 상속받아 class를 하나 생성하면 된다. 

```dart
class MyNavigatorObserver extends NavigatorObserver {
  @override
  void didPush(Route<dynamic> route, Route<dynamic>? previousRoute){
    log('did push route')
  }    
  
  @override
  void didPop(Route<dynamic> route, Route<dynamic>? previousRoute){
    log('did pop route')
  }
}
```

그리고 `MyNavigatorObserver`를 `GoRouter`에 넘겨준다.

```dart
GoRouter(
	...
  observers: [ // Add your navigator observers
    MyNavigatorObserver(),
  ]
)
```

이렇게 하면, event가 trigger될 때마다, navigator가 노티를 받게 된다. 



# References

* https://medium.com/@antonio.tioypedro1234/flutter-go-router-the-essential-guide-349ef39ec5b3 (위 글은 본 레퍼런스에서 발췌 및 번역함)

* https://www.kodeco.com/28987851-flutter-navigator-2-0-using-go_router (go_router 강의, 아직 확인하지는 못함.)
