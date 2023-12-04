---
title: DropdownButton in Flutter
date: 2023-11-14
tags: 
    - flutter
    - DropdownButton
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---



# Dropdown 버튼을 만들기 위한 위젯

```dart
// village 프로젝트에서의 사용 예제. 
DropdownButton(
    value: dropdownValue, // 최초에 보여지는 값
    elevation: 16,
    items: dropDown.map<DropdownMenuItem<String>>((String value) {
      return DropdownMenuItem<String>(
        value: value,
        child: Text(value),
      );
    }).toList(),
    onChanged: (String? value) {
      // This is called when the user selects an item.
      setState(() {
        dropdownValue = value!;
        int valueIdx = dropDown.indexOf(value);
        _firstGuardian = Guardian.values[valueIdx];
      });
    },
  ),
```

