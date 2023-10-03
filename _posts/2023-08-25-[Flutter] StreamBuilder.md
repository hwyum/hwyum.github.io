---
title: Flutter Stream
date: 2023-08-25
tags: 
    - flutter
    - stream
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

# StreamBuilder 구현하기

문제: 유저가 채팅 메시지를 보낼 때마다, 채팅 히스토리를 가져와서 화면에 보여줘야 한다. (채팅 버블)

StreamBuilder를 사용해 구현하고 있는데, 현재 받아오는 Stream은 다음과 같다. 

```dart
StreamBuilder<List<ChatModel>>(
        stream: (chatroomMateId != null)
            ? ChatService().getChatHistory(chatroomMateId, 0)
            : null,
        builder: (context, snapshot) { ... }
  )
```

`chatroomMateId`가 null이 아니면, `ChatService().getChatHistory(chatroomMateId, 0)`를 구독하게 된다. 

그리고, `ChatService().getChatHistory(chatroomMateId, 0)`는 아래와 같이 구현되어 있다. 

```dart
Stream<List<ChatModel>> getChatHistory(
      int chatroomMateId, int? sinceChatId) async* {
    Map<String, dynamic> response;
    List<ChatModel> chatHistory = [];

    try {
      response = await SendData.getRequest(
        url: '/chat/get-chat-mate-history-by-id',
        headerParams: {
          'Chatroom-Mate-Id': chatroomMateId,
          'Since-Chat-Id': (sinceChatId == null) ? 0 : sinceChatId,
        },
      );

      if (response['e_msg'] == 'OK') {
        for (Map<String, dynamic> chat in response['chats']) {
          ChatModel chatmodel = ChatModel.fromJson(chat);
          chatHistory.add(chatmodel);
        }
        yield chatHistory; // 받아온 데이터를 스트림으로 방출한다.
      } else {
        yield [];
      }
    } catch (e) {
      logger.e(e);
    }
  }
```

그런데 위의 방식은 초기에 한 번만 읽어와서 방출하는 방식이라고 한다. 

**유저가 메시지를 보낼 때마다 스트림을 업데이트하고, 다시 화면에 뿌려주려면 GPT가 아래와 같이 StreamController를 활용하도록 수정하라고 한다.** 

**[GPT가 제안한 사항]**

```dart
class ChatService {
  final StreamController<List<ChatModel>> _chatStreamController = StreamController<List<ChatModel>>();

  Stream<List<ChatModel>> get chatStream => _chatStreamController.stream;

  Future<void> getChatHistory(int chatroomMateId, int? sinceChatId) async {
    Map<String, dynamic> response;
    List<ChatModel> chatHistory = [];

    try {
      response = await SendData.getRequest(
        url: '/chat/get-chat-mate-history-by-id',
        headerParams: {
          'Chatroom-Mate-Id': chatroomMateId,
          'Since-Chat-Id': (sinceChatId == null) ? 0 : sinceChatId,
        },
      );

      if (response['e_msg'] == 'OK') {
        for (Map<String, dynamic> chat in response['chats']) {
          ChatModel chatmodel = ChatModel.fromJson(chat);
          chatHistory.add(chatmodel);
        }
        _chatStreamController.add(chatHistory); // 스트림에 데이터 추가
      } else {
        _chatStreamController.add([]); // 에러나 데이터 없는 경우에도 스트림에 빈 데이터 추가
      }
    } catch (e) {
      logger.e(e);
      _chatStreamController.addError(e); // 에러 발생 시 스트림에 에러 추가
    }
  }

  // Dispose 메서드 필요
  void dispose() {
    _chatStreamController.close();
  }
}

```

# 여러 Stream을 하나의 Stream으로 합치기

채팅 리스트를 조회할 때, 내가 요청한 채팅 리스트와 요청 받은 채팅 리스트를 각기 스트림으로 구독하고, 두 개의 스트림을 하나로 합칠 필요가 있다. (이것을 채팅탭에서 전체 리스트로 뿌려준다.)

