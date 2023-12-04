---

---

## Riverpod_generator

Riverpod_generator를 통해 @riverpod annotation으로 Privder를 정의할 수 있다. 

아래 예제에서 `fetchProducts` 함수를 통해 `fetchProductProvider`가 정의 된다.



(https://pub.dev/packages/riverpod_generator)

```dart
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:dio/dio.dart';

part 'my_file.g.dart';

// Using riverpod_generator, we define Providers by annotating functions with @riverpod.
// In this example, riverpod_generator will use this function and generate a matching "fetchProductProvider".
// The following example would be the equivalent of a "FutureProvider.autoDispose.family"
@riverpod
Future<List<Product>> fetchProducts(FetchProductRef ref, {required int page, int limit = 50}) async {
  final dio = Dio();
  final response = await dio.get('https://my-api/products?page=$page&limit=$limit');
  final json = response.data! as List;
  return json.map((item) => Product.fromJson(item)).toList();
}


// Now that we defined a provider, we can then listen to it inside widgets as usual.
Consumer(
  builder: (context, ref, child) {
    AsyncValue<List<Product>> products = ref.watch(fetchProductProvider(page: 1));

    // Since our provider is async, we need to handle loading/error states
    return products.when(
      loading: () => CircularProgressIndicator(),
      error: (err, stack) => Text('error: $err'),
      data: (products) {
        return ListView(
          children: [
            for (final product in products)
              Text(product.name),
          ],
        );
      },
    );
  },
);
```

