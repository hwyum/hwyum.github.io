---
title: "LangChain으로 LLM 활용하기"
date: 2023-06-14
tags: 
    - Python
    - LLM
    - LangChain
categories: 
    - ML
    - Python
    - LLM
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

# References

- LangChain Python 공식 [Documentations](https://python.langchain.com/docs/get_started/introduction.html)
- [Getting Started with LangChain: A Beginner’s Guide to Building LLM-Powered Applications](https://medium.com/towards-data-science/getting-started-with-langchain-a-beginners-guide-to-building-llm-powered-applications-95fc8898732c)

![image-20230614230349387](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230614230349387.png)

# [LangChain](https://github.com/hwchase17/langchain) 이란 무엇인가?

LLM 기반 어플리케이션 개발을 쉽게 할 수 있도록 하는 프레임워크로 아래와 같은 기능을 제공한다. 

- 다양한 foundation model들을 사용할 수 있는 범용적인 인터페이스 제공
- 프롬프트를 매니지 할 수 있도록 도와주는 프레임워크
- Language model이 다른 소스의 데이터와 연결될 수 있도록 지원 (Data-aware)
- long-term memory, external data, 다른 LLM들, LLM이 다루지 못하는 다른 agent 들 (예. 계산기, 검색) 에 대한 중앙 인터페이스 (language model이 환경과 인터액션 가능)



## LangChain 설치하기

```python
pip install langchain
```

## API Keys

### LLM Provider (required)

- LLM Provider 에는 소유권이 있는 모델과 오픈소스로 활용할 수 있는 모델이 존재한다. 
  - <img src="https://miro.medium.com/v2/resize:fit:1400/1*-qwmPSD82bd3k7y6exwWkA.png" alt="LLM Providers: Proprietary and open-source foundation models" style="zoom:67%;" />
  - 오픈 소스 모델은 모델 사이즈가 유료 모델 대비 작지만 비용측면에서 효율적일 수 있다. 
    - [BLOOM](https://huggingface.co/bigscience/bloom) by BigScience
    - [LLaMA](https://huggingface.co/docs/transformers/main/en/model_doc/llama) by Meta AI
    - [Flan-T5](https://huggingface.co/google/flan-t5-xl) by Google
    - [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b) by Eleuther AI
  - 많은 오픈 소스 모델들은 [Hugging Face](https://huggingface.co/)를 Hub로 호스팅 되고 있다. Hugging Face API Key를 얻으려면, Hugging Face 계정과, Token을 생성해야 한다. ([Access Tokens](https://huggingface.co/settings/tokens))

### Vector Database (Optional)

- 특정 벡터 DB를 사용하고자 한다면, API 키가 필요하고, 가격 정책을 확인해야 한다. 
  - [Pinecone](https://www.pinecone.io/), [Weaviate](https://weaviate.io/), or [Milvus](https://milvus.io/),
- Faiss의 경우 sign-up을 필요로 하지 않는다.

### Tools (Optional) 

- Tool을 활용하면 LLM을 외부 application과 소통할 수 있게 할 수 있다. (예. OpenWeatherMap, SerpAPI)
- 이를 위해 API Key등록이 필요하고, 가격 정책을 확인해야 한다. 

## LangChain이 제공하는 모듈

#### [Model I/O](https://python.langchain.com/docs/modules/model_io/)

Interface with language models

#### [Data connection](https://python.langchain.com/docs/modules/data_connection/)

Interface with application-specific data

#### [Chains](https://python.langchain.com/docs/modules/chains/)

Construct sequences of calls

#### [Agents](https://python.langchain.com/docs/modules/agents/)

Let chains choose which tools to use given high-level directives

#### [Memory](https://python.langchain.com/docs/modules/memory/)

Persist application state between runs of a chain

#### [Callbacks](https://python.langchain.com/docs/modules/callbacks/)

Log and stream intermediate steps of any chain

# 1. Model I/O

언어 모델을 활용한 어플리케이션의 핵심은 "모델". Langchain은 어떠한 언어모델과도 interface할 수 있는 모듈을 제공한다. 

- Prompt: Templatetize, dynamically select, and manage model inputs
- Language Models : Make calls to language models through common interfaces
- Output Parsers: Extract Informatio from model outputs

![model_io_diagram](https://python.langchain.com/assets/images/model_io-1f23a36233d7731e93576d6885da2750.jpg)

## (1) Prompt: Managing LLM inputs

Prompt 는 모델에 주입되는 '인풋'을 의미하고, 이러한 인풋은 여러 컴포넌트로부터 생성될 수 있다. (예를 들어, 유저가 무언가를 선택함에 따라 다른 인풋을 줄 수도 있을 것) LangChain은 `PromptTemplates`를 제공하고, `PromptTemplate` 이러한 인풋 생성을 책임진다. 

LangChain은 Propmt에 관해 아래와 같은 밸류를 제공한다.

- string prompt와 message prompt에 대한 표준화된 인터페이스
- string prompt templates와 message prompt tempate에 대한 표준화된 인터페이스
- Example Selectors: prompt에 예제들을 주입할 수 있도록 하는 메소드
- OutputParsers: language model이 정보를 출력해야 하는 형식으로 지시문을 삽입하는 방법 및 그 문자열 출력을 포맷으로 파싱하는 방법에 대한 메서드. 

### PromptTemplates

prompt template는 프롬프트를 생성하는 재생가능한 방법을 제공한다. Text string과, 엔드 유저가 프롬프트를 생성할 수 있도록 하는 매개변수들을 받을 수 있다. 

프롬프트는 아래의 내용을 포함할 수 있다:

- 언어모델에 대한 instruction
- 언어모델이 더 나은 결과를 생성할 수 있게 도와주는 few shot 예제들의 집합
- 언어 모델에 대한 질문

간단한 예는 다음과 같다. 

```python
from langchin import PromptTemplate

template = """/
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate.from_template(template)
prompt.format(product="colorful socks")
```

```
I want you to act as a naming consultant for new companies.
What is a good name for a company that makes colorful socks?
```

#### Few-shot prompt templates

##### Using an exmple set

1. Exmple을 주입하기 위한 예제 세트 생성

```python
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
  {
    "question": "Who lived longer, Muhammad Ali or Alan Turing?",
    "answer": 
"""
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
"""
  },
  {
    "question": "When was the founder of craigslist born?",
    "answer": 
"""
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
"""
  },
  {
    "question": "Who was the maternal grandfather of George Washington?",
    "answer":
"""
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
"""
  },
  {
    "question": "Are both the directors of Jaws and Casino Royale from the same country?",
    "answer":
"""
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
"""
  }
]
```

2. formatter 생성

   - formatter는 `PromptTemplate` object여야 한다. 

   ```python
   example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")
   
   print(example_prompt.format(**examples[0]))
   ```

   ```
       Question: Who lived longer, Muhammad Ali or Alan Turing?
       
       Are follow up questions needed here: Yes.
       Follow up: How old was Muhammad Ali when he died?
       Intermediate answer: Muhammad Ali was 74 years old when he died.
       Follow up: How old was Alan Turing when he died?
       Intermediate answer: Alan Turing was 41 years old when he died.
       So the final answer is: Muhammad Ali
   ```

3. `FewShotPromptTemplate` 오브젝트 생성

   - 해당 오브젝트는 few shot 예제들과 포매터를 인풋으로 받는다. 

   ```python
   prompt = FewShotPromptTemplate(
       examples=examples, 
       example_prompt=example_prompt, 
       suffix="Question: {input}", 
       input_variables=["input"]
   )
   
   print(prompt.format(input="Who was the father of Mary Ball Washington?"))
   ```

   ```
       Question: Who lived longer, Muhammad Ali or Alan Turing?
       
       Are follow up questions needed here: Yes.
       Follow up: How old was Muhammad Ali when he died?
       Intermediate answer: Muhammad Ali was 74 years old when he died.
       Follow up: How old was Alan Turing when he died?
       Intermediate answer: Alan Turing was 41 years old when he died.
       So the final answer is: Muhammad Ali
       
       
       Question: When was the founder of craigslist born?
       
       Are follow up questions needed here: Yes.
       Follow up: Who was the founder of craigslist?
       Intermediate answer: Craigslist was founded by Craig Newmark.
       Follow up: When was Craig Newmark born?
       Intermediate answer: Craig Newmark was born on December 6, 1952.
       So the final answer is: December 6, 1952
       
       
       Question: Who was the maternal grandfather of George Washington?
       
       Are follow up questions needed here: Yes.
       Follow up: Who was the mother of George Washington?
       Intermediate answer: The mother of George Washington was Mary Ball Washington.
       Follow up: Who was the father of Mary Ball Washington?
       Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
       So the final answer is: Joseph Ball
       
       
       Question: Are both the directors of Jaws and Casino Royale from the same country?
       
       Are follow up questions needed here: Yes.
       Follow up: Who is the director of Jaws?
       Intermediate Answer: The director of Jaws is Steven Spielberg.
       Follow up: Where is Steven Spielberg from?
       Intermediate Answer: The United States.
       Follow up: Who is the director of Casino Royale?
       Intermediate Answer: The director of Casino Royale is Martin Campbell.
       Follow up: Where is Martin Campbell from?
       Intermediate Answer: New Zealand.
       So the final answer is: No
       
       
       Question: Who was the father of Mary Ball Washington?
   ```

## (2) Models 

**다양한 LLM과 embedding model들을 선택할 수 있다.** LangChain은 다양한 종류의 모델에 대해 표준화된 인터페이스를 제공하며, 모델의 종류에 따라 다른 형태의 input과 ouput이 존재한다.

### Language Models

- Langauge Model에 2개의 sub-type이 존재한다. 

  - LLMs: 텍스트를 인풋으로 받고 텍스트를 리턴하는 API를 랩핑.
  - ChatModels: 챗 메시지들을 인풋으로 받고, 챗 메시지를 리턴하는 모델을 랩핑. 

  - OpenAI에서 제공하는 LLM과 ChatModel을 랩핑한 클래스를 살펴보자.

    ```python
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    
    >>> llm = OpenAI()
    >>> chat_model = ChatOpenAI()
    
    # text -> text interface
    >>> llm.predict("say hi!")
    '\n\nHi there!'
    
    >>> chat_model.predict("say hi!")
    'Hello there!'
    
    # messages -> message interface
    >>> llm.predict_messages([HumanMessage(content="say hi!")])
    AIMessage(content='\n\nHi there!', additional_kwargs={}, example=False)
    
    >>> chat_model.predict_messages([HumanMessage(content="say hi!")])
    AIMessage(content='Hi there! How may I assist you today?', additional_kwargs={}, example=False)
    
    ```

    

#### LLMs

```python
# Proprietary LLM from e.g. OpenAI
# pip install openai
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")

# # Alternatively, open-source LLM hosted on Hugging Face
# # pip install huggingface_hub
# from langchain import HuggingFaceHub
# llm = HuggingFaceHub(repo_id = "google/flan-t5-xl")

# The LLM takes a prompt as an input and outputs a completion
prompt = "Alice has a parrot. What animal is Alice's pet?"
completion = llm(prompt)

print(f'{prompt}\n{completion}')
```

```
Alice has a parrot. What animal is Alice's pet?
Alice's pet is a parrot.
```



#### [Chat models](https://python.langchain.com/docs/modules/model_io/models/chat/)

- LLM과 유사하게 LLM Provider가 제공하는 chat model을 랩핑한다. 

  ```python
  from langchain.chat_models import ChatOpenAI
  
  chat = ChatOpenAI()
  ```

- chat message들의 리스트를 input으로 받고, chat message를 리턴한다.

  LangChain에서 제공하는 메시지 타입으로 `AIMessage`, `HumanMessage`, `SystemMessage`, `ChatMessage` 가 존재하고, `ChatMessage` 는 임의의 역할을 파라메터로 받는다. 

- Chat model에 한개 혹은 그 이상의 메시지들을 전달하고, chat model 객체를 `__call__`하면 메시지 응답을 받을 수 있다. 

  ```python
  # Messages in -> Message out
  from langchain.schema import (
      AIMessage,
      HumanMessage,
      SystemMessage
  )
  
  chat([HumanMessage(content="Translate this sentence from English to French: I love programming.")])
  ```

  ```
      AIMessage(content="J'aime programmer.", additional_kwargs={})
  ```

  ```python
  # multiple messages -> message
  messages = [
      SystemMessage(content="You are a helpful assistant that translates English to French."),
      HumanMessage(content="I love programming.")
  ]
  chat(messages)
  ```

  ```
      AIMessage(content="J'aime programmer.", additional_kwargs={})
  ```

`Generate`를 사용하면, batch 로 메시지들을 전달할 수 있다. 결과로 `LLMResult`를 반환하고, `message`파라메터를 가진다. 

```python
# generate for batch messges
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to Korean."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to Korean."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result
```

###### ![image-20230620083330795](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230620083330795.png)



### Text embedding models

- Text를 인풋으로 받고, `list of floats (embeddings)`를 리턴한다. 임베딩은 text의 수치적인 표현 방식을 의미한다. (이후에 text간 유사성 계산 등에 활용될 수 있다.)

```python
# Proprietary text embedding model from e.g. OpenAI
# pip install tiktoken
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# # Alternatively, open-source text embedding model hosted on Hugging Face
# # pip install sentence_transformers
# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# The embeddings model takes a text as an input and outputs a list of floats
text = "Alice has a parrot. What animal is Alice's pet?"
text_embedding = embeddings.embed_query(text)
```

```
[0.012726864777505398,
 -0.009588395245373249,
 -0.007531091570854187,
 -0.012127590365707874,
 -0.012961632572114468,
 ...]
```

## (3) [Output parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)

언어모델이 출력하는 문자열을 구조화 할 필요성이 존재한다. Output parser는 이러한 역할을 담당한다. Output parser가 반드시 구현해야 하는 두 가지 메서드는 다음과 같다:

- `Get format instructions`: 언어 모델의 아웃풋이 어떻게 formatted되어야 하는지에 대한 명령(instructions)을 포함하는 문자열을 리턴하는 메서드
- `Parse`: (언어모델의 응답으로 예상되는) 문자열을 받고, 이를 특정 구조로 파싱하는 메서드

다음은 output parser의 main type인 `PydanticOutputParser`에 대한 예시이다. 
`Pydantic (JSON) parser`는 유저가 JSON Schema를 임의로 정의할 수 있게 해주고, 이러한 스키마를 만족하는 JSON 아웃풋을 출력하도록 LLM에 질의한다. 

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

model_name = 'text-davinci-003'
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

# Define your desired data structure.
class Joke(BaseModel):
  setup: str = Field(description="question to set up a joke")
  punchline: str = Field(description="answer to resolve the joke")
  
  # You can add custom validation logic easily with Pydantic.
  @validator("setup")
  def question_ends_with_question_mark(cls, field):
    if field[-1] != "?":
      raise ValueError("Badly formed question!")
    return field
  
# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=joke_query)
```

```python
print(_input)
```

```
text='Answer the user query.\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"setup": {"title": "Setup", "description": "question to set up a joke", "type": "string"}, "punchline": {"title": "Punchline", "description": "answer to resolve the joke", "type": "string"}}, "required": ["setup", "punchline"]}\n```\nTell me a joke.\n'
```

```python
output = model(_input.to_string())

parser.parse(output)
```

```
Joke(setup='Why did the chicken cross the road?', punchline='To get to the other side!')
```

이 외에도, List parser, Datetime parser, Enum parser, Auto-fixting ㅔarser 등 여러 종류의 parser를 제공한다.

# 2. Data connection

많은 LLM Application은 모델의 학습 데이터셋이 아닌 user-specific data를 필요로 한다. LangChain은 Load, Transform, 유저 데이터 조회를 위한 빌딩 블록들을 제공한다. 

- Document loaders: 여러 소스로부터 다큐먼트들을 로드한다.
- Document transformers: Split documents, drop redundant documents, and more
- Text embedding models: Take unstrcutured text and turn it into a list of floating point numbers
- Vector stores: Store and search over embedded data
- Retrievers: Query your data

![data_connection_diagram](https://python.langchain.com/assets/images/data_connection-c42d68c3d092b85f50d08d4cc171fc25.jpg)

## Document loaders

데이터를 source로부터 `Document`로 로드하기 위해 사용한다. `Document`는 글조각이자 metadata와 연관되어 있다. 에를 들어, 웹페이지의 컨텐츠나, 유튜브 비디오의 스크립트를 로딩하기 위해 단순하게 `.TXT`파일을 로딩하기 위한 document loader가 있을 수 있다. 

데이터를 lazy하게 메모리에 로드하기 위해 'lazy load'도 구현할 수 있는 옵션을 제공한다. 

가장 단순한 형태의 로더는 하나의 파일을 텍스트로 읽고, 이를 하나의 Document로 위치시킨다. 

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("./index.md")
loader.load()
```

```
[
    Document(page_content='---\nsidebar_position: 0\n---\n# Document loaders\n\nUse document loaders to load data from a source as `Document`\'s. A `Document` is a piece of text\nand associated metadata. For example, there are document loaders for loading a simple `.txt` file, for loading the text\ncontents of any web page, or even for loading a transcript of a YouTube video.\n\nEvery document loader exposes two methods:\n1. "Load": load documents from the configured source\n2. "Load and split": load documents from the configured source and split them using the passed in text splitter\n\nThey optionally implement:\n\n3. "Lazy load": load documents into memory lazily\n', metadata={'source': '../docs/docs_skeleton/docs/modules/data_connection/document_loaders/index.md'})
]
```

### CSV

```python
# Load CSV data with a single row per document.
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='..example_data/mlb_teams_2012.csv')
data = loader.load()

print(data)
```

```
    [Document(page_content='Team: Nationals\n"Payroll (millions)": 81.34\n"Wins": 98', lookup_str='', metadata={'source': './example_data/mlb_teams_2012.csv', 'row': 0}, lookup_index=0), Document(page_content='Team: Reds\n"Payroll (millions)": 82.20\n"Wins": 97', lookup_str='', metadata={'source': './example_data/mlb_teams_2012.csv', 'row': 1}, lookup_index=0), Document(page_content='Team: Yankees\n"Payroll (millions)": 197.96\n"Wins": 95', lookup_str='', metadata={'source': './example_data/mlb_teams_2012.csv', 'row': 2}, lookup_index=0), Document(page_content='Team: Giants\n"Payroll (millions)": 117.62\n"Wins": 94', lookup_str='', metadata={'source': './example_data/mlb_teams_2012.csv', 'row': 3}, lookup_index=0), ...]
```

### File Directory

하나의 디렉토리에 있는 모든 다큐먼트를 모두 로드하는 방법을 다룬다. 

```python
from langchain.document_loaders import DirectoryLoader

# 어떤 파일을 로딩할 것인지에 관련해서, `glob` 파라메터를 활용할 수 있다. 
loader = DirectoryLoader('../', glob="**/*.md")

docs = loader.load()

len(docs)
```

```
1
```

### HTML

```python
from langchain.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader("example_data/fake-content.html")
data = loader.load()
data
```

```
    [Document(page_content='My First Heading\n\nMy first paragraph.', lookup_str='', metadata={'source': 'example_data/fake-content.html'}, lookup_index=0)]
```

### JSON

### PDF

## Document transformers

데이터를 로드하게 되면, 애플리케이션에 적절한 형태로 변환해야 한다. 가장 단순한 예시는, 긴 문서를 작은 청크로 쪼개서 모델의 컨텍스트 윈도우에 맞도록 하는 것이다. LangChain은 split, combie, filter 등의 빌트인 document transformer들을 제공한다. 

### Text splitters

```python
# This is a long document we can split up.
with open('../../state_of_the_union.txt') as f:
    state_of_the_union = f.read()
    
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])
```

```
    page_content='Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and' metadata={'start_index': 0}
    page_content='of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.' metadata={'start_index': 82}
```

## Vector stores

벡터 DB를 활용해서 임베딩 형태의 벡터를 저장하고 쿼리할 수 있도록 한다. 

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


raw_documents = TextLoader('/content/drive/MyDrive/romeo_and_juliet.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

db = FAISS.from_documents(documents, OpenAIEmbeddings())
```

### Similarity search

```python
query = "juliet pretended to be dead"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

![image-20230620090202049](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230620090202049.png)

# 3. Chains

간단한 애플리케이션에서는 단일 LLM을 사용하는 것이 괜찮지만, 더 복잡한 애플리케이션은 다른 LLM들이나 다른 컴포넌트와 연결된 LLM들이 필요하다. LangChain은 이러한 "연결된" 애플리케이션을 위한 Chain 인터페이스를 제공한다. 

체인을 사용하면 여러 컴포넌트를 결합하여 하나의 일관된 애플리케이션을 만들 수 있다. 예를 들어, 사용자 입력을 받아 PromptTemplate로 포맷하고, 그 포맷된 응답을 LLM에 전달하는 체인을 만들 수 있다. 우리는 여러 체인을 결합하거나, 다른 컴포넌트와 체인을 결합함으로써 더 복잡한 체인을 만들 수 있다. 

`LLMChain`은 가장 기본적인 빌딩 블록 체인이다. 이것은 프롬프트 템플릿을 가져와서 사용자 입력으로 포맷하고, LLM에서의 응답을 반환한다. 

`LLMChain`을 이용하기 위해서 먼저 prompt template을 생성한다. 

```python
from langchain.llms import OpenAI
from langchain.propmpts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
	input_variables=["product"],
  template="What is a good name for a company that makes {product}?",
)

# 이제 사용자 입력을 받아서 프롬프트를 포맷하고, 그것을 LLM에 전송하는 매우 간단한 체인을 만들 수 있다. 
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))

```

```
    Colorful Toes Co.
```

## LLMChain

LLMChain은 language model에 일부 기능을 더하는 간단한 체인이다. 다른 체인 및 에이전트를 포함하여 LangChain 전체에서 널리 사용된다. 

```python
# Get Started
from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "What is a good name for a company that makes {product}?"

llm = OpenAI(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
llm_chain("colorful socks")
```

```
    {'product': 'colorful socks', 'text': '\n\nSocktastic!'}
```

`__call__`과 `run`메서드 외에 체인 로직을 호출하는 다른 방법도 제공한다. 

- `apply`는 list of input에 대해 체인을 작동시킨다. 
- `generate`는` apply`와 유사한데, string대신에 `LLMResult`를 리턴한다. `LLMResult`는 token usage와 finish reason과 같은 유용한 정보를 제공한다. 
- `predict`는 `run`과 유사한데, input key들이 딕셔너리가 아닌 keyword arguemnt로 명시적으로 넘겨주는 형태를 취한다. 

### Parsing the outputs

`LLMChain`은 기본적으로 parsing을 제공하지 않는다. output parser를 LLM Output에 적용하려면, `predict`, `apply` 대신에 `predict_and_parse`, `apply_and_parse`를 활용하면 된다. 

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
template = """List all the colors in a rainbow"""
prompt = PromptTemplate(template=template, input_variables=[], output_parser=output_parser)
llm_chain = LLMChain(prompt=prompt, llm=llm)

llm_chain.predict_and_parse()
```

```
    ['Red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
```

## Router

주어진 input을 고려할 때, 다음 chain을 동적으로 선택하기 위한 체인을 만들기 위한 `RoutherChain`패러다임을 사용하기 위한 방법을 나타낸다. 

Router chain은 두 개의 컴포넌트로 구성됨:

- RouterChain 그 자체 (어떤 체인을 호출할 것인지 선택)
- destination_chains: router chain이 라우팅할 수 있는 대상 체인

다음 예시는 주어진 질문에 대해서 어떤 프롬프트가 적절한지 선택하고 해당 프롬프트를 활용해서 답변하는question-answering chain을 생성는 `MultiPromptChain`에서 routing chain이 사용되는 예시이다. 

```python
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
]

llm = OpenAI()

destination_chains = {}
for p_info in prompt_infos:
  name = p_info["name"]
  prompt_template = p_info["prompt_template"]
  prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
  chain = LLMChain(llm=llm, prompt=prompt)
  destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")
```

### LLMRouterChain

이 체인은 어떻게 라우팅 할 것인지를 결정하기 위해 LLM을 사용한다.

```python
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

print(chain.run("What is black body radiation?"))
```

```
    
    
    > Entering new MultiPromptChain chain...
    physics: {'input': 'What is black body radiation?'}
    > Finished chain.
    
    
    Black body radiation is the term used to describe the electromagnetic radiation emitted by a “black body”—an object that absorbs all radiation incident upon it. A black body is an idealized physical body that absorbs all incident electromagnetic radiation, regardless of frequency or angle of incidence. It does not reflect, emit or transmit energy. This type of radiation is the result of the thermal motion of the body's atoms and molecules, and it is emitted at all wavelengths. The spectrum of radiation emitted is described by Planck's law and is known as the black body spectrum.
```

```python
print(
    chain.run(
        "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3"
    )
)
```

```
    > Entering new MultiPromptChain chain...
    math: {'input': 'What is the first prime number greater than 40 such that one plus the prime number is divisible by 3'}
    > Finished chain.
    ?
    
    The answer is 43. One plus 43 is 44 which is divisible by 3.
```

## Sequential

하나의 콜에서의 아웃풋을 다시 인풋으로 활용하고자 할 때 유용하다. 시퀀셜 체인은 여러 체인을 연결하고, 이를 파이프라인으로 구성해서, 구체적인 시나리오를 수행하게 할 수 있다. 

시퀀셜 체인에는 두 가지 타입이 존재한다. 

- `SimpleSequentialChain`: 단순한 형태의 sequential chain으로, 각 단계는 하나의 input/output으로 구성되고, 해당 output이 다음 스텝의 input이 된다.
- `SequentialChain`: sequential chain의 일반적인 형태로, 여러개의 input/output을 허용한다. 

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# This is an LLMChain to write a synopsis given a title of a play.
llm = OpenAI(temperature=.7)
template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is an LLMChain to write a review of a play given a synopsis.
llm = OpenAI(temperature=.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)


# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)
```

![image-20230619104112290](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619104112290.png)

### SequentialChain

input과 output이 여러개인 경우에 대한 예시를 살펴본다. 이 경우에는 input/output 변수명에 특히 신경을 써야 한다. 

```python
# This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
llm = OpenAI(temperature=.7)
template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title", 'era'], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

# This is an LLMChain to write a review of a play given a synopsis.
llm = OpenAI(temperature=.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SequentialChain
overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True)


```

![image-20230619104926683](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619104926683.png)

### Memory in Sequential Chains

각 스텝의 체인에서 어떤 컨텍스트를 전달할 필요가 있을 수 있다. `SimpleMemory`를 활용하면 이를 편하게 관리할 수 있다. 예를 들어서, 위의 `SequentialChain`예시에서 연극의 일자, 시간, 장소 정보를 포함시킬 수 있고, 만들어진 시놉시스와 리뷰를 활용해서 social media post text를 생성할 수 있다. 이것들을 `input_variable`로 포함시킬 수도 있고, `SimpleMemory`를 활용할 수도 있다. 

```python
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory

llm = OpenAI(temperature=.7)
template = """You are a social media manager for a theater company.  Given the title of play, the era it is set in, the date,time and location, the synopsis of the play, and the review of the play, it is your job to write a social media post for that play.

Here is some context about the time and location of the play:
Date and Time: {time}
Location: {location}

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:
{review}

Social Media Post:
"""
prompt_template = PromptTemplate(input_variables=["synopsis", "review", "time", "location"], template=template)
social_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="social_post_text")

overall_chain = SequentialChain(
  memory=SimpleMemory(memories={"time": "December 25th, 8pm KST", "location": "대학로 나무극장"}),
    chains=[synopsis_chain, review_chain, social_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["social_post_text"],
    verbose=True
)

```

![image-20230619105836539](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619105836539.png)

## Transformation

아래는 매우 긴 텍스트를 받아서, 첫번째 3개의 문단으로 필터링 이후 LLMChain에 그것을 요약할 수 있도록 넘겨주는 dummy transformation을 생성하는 예시이다. 

```python
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

with open("/content/drive/MyDrive/romeo_and_juliet.txt") as f:
    romeo_and_juliet = f.read()
    
def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    return {"output_text": shortened_text}


transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=transform_func
)

template = """Summarize this text:

{output_text}

Summary:"""
prompt = PromptTemplate(input_variables=["output_text"], template=template)
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])
```

![image-20230619111701447](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619111701447.png)

## Documents

문서들을 다루기 위한 core chain들이 존재한다. 문서 요약, 문서에 대한 Q&A, 문서로부터 정보 추출 등을 하는 경우 유용하다. 

### Stuff

Stuff documents chain은 가장 직관적인 형태의 document chain으로 doucment 리스트를 받고, 그것들을 몽땅 propmt에 넣어서, LLM으로 전달한다. 문서들이 크기가 작고, 몇 개의 문서만 전달되는 경우에 유용하다. 

![image-20230619112101708](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619112101708.png)

### Refine

Input document들을 순환하면서, iterative하게 답변을 업데이트하는 방식으로 답변을 생성한다. 각각의 문서에서 non-document inputs, current document, 가장 최신의 중간 답변을 LLM에 전달해서 새로운 답변을 얻는다. 

![refine_diagram](https://python.langchain.com/assets/images/refine-a70f30dd7ada6fe5e3fcc40dd70de037.jpg)

### Map reduce

map reduce documents chain은 처음에 LLN 체인을 각 document에 개별적으로 적용해서, 각 체인의 아웃풋을 새로운 document로 취급한다. (Map Step) 그리고 모든 새로운 document들을 별도의 combine documents chain에 넘겨서 하나의 single output을 얻도록 한다. (Reduce step)

![map_reduce_diagram](https://python.langchain.com/assets/images/map_reduce-c65525a871b62f5cacef431625c4d133.jpg)

### Map re-rank

map re-rank documents chain은 최초의 프롬프트를 각 document에 적용하고, task 수행을 완료할 뿐만 아니라, 그 응답이 얼만 ㅏ확실한지에 대한 스코어까지 부여한다. 가장 스코어가 높은 답변이 리턴된다. 

![map_rerank_diagram](https://python.langchain.com/assets/images/map_rerank-0302b59b690c680ad6099b7bfe6d9fe5.jpg)

### 그 외에 지원하고 있는 체인이 많다... 

#### API Chains

APIChain은 LLM이 정보를 받아오기 위해 API를 사용할 수 있도록 지원한다. ([체이닝이 가능한 API](https://github.com/hwchase17/langchain/tree/master/langchain/chains/api)가 한정되어 있는 것 같다. )

#### Retrieval QA

문서에 대해 정보를 받아오기 위한 체인으로 LLM과 vector score을 연결한다. 

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

loader = TextLoader("/content/drive/MyDrive/romeo_and_juliet.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
  llm=OpenAI(), 
  chain_type="stuff", 
  retriever=docsearch.as_retriever()
) # RetrievalQA를 활용해 LLM - docsearch 연결
```

![image-20230619150753062](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619150753062.png)

optional parameter를 활용해 응답에 활용한 source document를 함께 리터할 수도 있다. 

```python
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
```

![image-20230619151135315](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619151135315.png)

#### [Conversational Retrieval QA](https://python.langchain.com/docs/modules/chains/popular/chat_vector_db)

RetrievalQAChain에 채팅 히스토리를 제공하기 위한 체인이다. 

#### [SQL](https://python.langchain.com/docs/modules/chains/popular/sqlite)

SQL DB에 대해 Q&A를 하기 위해 `SQLDatabaseChain`을 사용할 수 있다. 

#### Summarization

여러개의 document들을 요약하기 위해 사용될 수 있는 체인이다. 한 가지 방법으로, 여러개의 작은 도큐먼트를 인풋으로 제공하고, 그것들을 chunk로 쪼개고, MapReduceDocumentsChain을 적용한다. 요약이 StuffDocumentChain이 되는 대신에, RefineDocumentsChain으로 선택할 수 있다. 

```python
# Prepare Data
# 본 예제에서는 하나의 긴 텍스트를 쪼개서 여러 개의 문서로 만듬. (실제로는 여러 개의 문서 로딩 가능)

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter()

with open("/content/drive/MyDrive/romeo_and_juliet.txt") as f:
    romeo_and_juliet = f.read()
texts = text_splitter.split_text(romeo_and_juliet)

from langchain.docstore.document import Document
docs = [Document(page_content=t) for t in texts[:3]]
```

```python
# Quickstart
# If you just want to get started as quickly as possible, this is the recommended way to do it:

from langchain.chains.summarize import load_summarize_chain

chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)
```

![image-20230619152928124](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619152928124.png)

##### The `Stuff`Chain

`Stuff` 체인을 이용해서 요약하기 위한 방법이다. 

```python
chain = load_summarize_chain(llm, chain_type="stuff")
chain.run(docs)
```

```python
# Custom Prompts
# you can also use your own prompts with this chain. In this example, we will respond in Korean.

prompt_template = """Write a concise summary of the following:

{text}

CONCISE SUMMARY IN JAPANESE:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
chain.run(docs)
```

![image-20230619153754047](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619153754047.png)

##### The `map_reduce` Chain

Optional 파라메터를 통해 intermediate_step에 대해서도 리턴을 받을 수 있다. 

```python
chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce", return_intermediate_steps=True)

chain({"input_documents": docs}, return_only_outputs=True)
```

### Extraction

Extraction chain은  documnet로 부터 entity들을 추출하기 위한 스키마를 특정하기 위해 OpenAI functions parater를 활용한다. 이는 모델이 정확히 우리가 원하는 entity들의 스키마와 특성들을 추출할 수 있도록 돕는다. 

Extraction chain은 같은 passage에서, 여러 entity들을 추출하고자 할때 사용될 수 있다. 

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Extracting entities
# 엔티티 추출을 위해서 스키마를 생성해야 한다. 
schema = {
    "properties": {
        "person_name": {"type": "string"},
        "person_height": {"type": "integer"},
        "person_hair_color": {"type": "string"},
        "dog_name": {"type": "string"},
        "dog_breed": {"type": "string"},
    },
    "required": ["person_name", "person_height"],
}

inp = """
Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde.
Alex's dog Frosty is a labrador and likes to play hide and seek.
        """

chain = create_extraction_chain(schema, llm)
```

![image-20230619172340879](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619172340879.png)

##### Pydantic example

요구되는 특성과 타입을 선택하기 위해 Pydantic schem를 활용할 수도 있다. `create_extraction_chain_pydantic`함수를 활용해서, Pydantic schema를 인풋으로 보내고, output은 우리가 원하는 스키마를 가지는 초기화된 객체로 반환된다. 

```python
from typing import Optional, List
from pydantic import BaseModel, Field

class Properties(BaseModel):
  person_name: str
  person_height: int
  person_hair_color: str
  dog_breed: Optional[str]
  dog_name: Optional[str]

chain = create_extraction_chain_pydantic(pydantic_schema=Properties,
                                         llm=llm)

inp = """
Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde.
Alex's dog Frosty is a labrador and likes to play hide and seek.
        """
```

# 4. Memory

기본적으로 체인과 에이전트들은 stateless이다. 그런데 챗봇 같은 애플리케이션에서는 이전 인터액션을 기억하는 게 필수적이다. Memory 클래슨느 이를 지원하기 위해 존재한다. 

LangChain은 메모리 컴포넌트를 두 가지 형태로 제공한다. 첫 번째로, 이전 챗 메시지들을 관리하고 조직하기 위한 helper utility들을 제공한다. 두 번재로, LangChain은 이러한 유틸리티들을 체인으로 합칠 수 있는 쉬운 방법들을 제공한다. 

## Get Started

일반적으로, 이전 대화의 시퀀스로부터 정보를 추출하는 함수들이 있고, 이후에 이 메모리를 체인에서 활용할 수 있을 것이다. 

메모레는 여러 종류의 정보들을 리턴할 수 있을 것이고 (최근 N개의 메시지, 이전 메시지들에 대한 요약 등등) 리턴된 정보들은 문자열이거나, 메시지들의 리스트들일 수 있다. 

아래에서는 가장 심플한 형태의 메모리인 "buffer" 메모리를 살펴볼 것인데, 이는 단순하게 이전에 모든 메시지들을 버퍼로 유지하는 것이다. 

## ChatMessageHistory

`ChatMessageHistory` 클래스가 가장 기반이 되는 유틸리티 클래스이다. Human messages, AI messages 들을 저장하고, 검색하는 편리한 메서드를 제공하는 매우 가벼운 래퍼이다. 

체인 바깥에서 메모리를 직접 관리하고자 할 때 이 클래스를 직접 활용할 수 있다. 

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("hi!")
history.add_ai_message("whats up?")

history.messages
```

```
    [HumanMessage(content='hi!', additional_kwargs={}),
     AIMessage(content='whats up?', additional_kwargs={})]
```

## ConversationBufferMemory

이제 이 간단한 개념을 체인에서 사용하는 방법이다. `ConversationBufferMemory`는 ChatMessageHistory를 감싸고 있으며 변수에서 메시지를 추출한다.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

memory.load_memory_variables({})
```

```
    {'history': 'Human: hi!\nAI: whats up?'}
```

히스토리를 메시지들의 리스트로 얻을 수도 있다. 

```python
memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

memory.load_memory_variables({})
```

```
    {'history': [HumanMessage(content='hi!', additional_kwargs={}),
      AIMessage(content='whats up?', additional_kwargs={})]}
```

## Using in a chain

```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(
	llm=llm,
  verbose=True,
  memory=ConversationBufferMemory()
)

conversation.predict(input="Hi there!")
```

![image-20230619182654826](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619182654826.png)

```python
conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
```

![image-20230619183025064](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619183025064.png)

## Saving Message History

메시지를 저장하고, 다시 로딩할 필요가 존재할 수 있다. 먼저 메시지들을 python dictionary로 전환하고, 이를 저장하고 로드하도록 할 수 있다. 

```python
import json

from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

history = ChatMessageHistory()

history.add_user_message("hi!")
history.add_ai_message("whats up?")

dicts = messages_to_dict(history.messages)

dicts
```

```
    [{'type': 'human', 'data': {'content': 'hi!', 'additional_kwargs': {}}},
     {'type': 'ai', 'data': {'content': 'whats up?', 'additional_kwargs': {}}}]
```

```python
new_messages = messages_from_dict(dicts)
new_mssages
```

```
    [HumanMessage(content='hi!', additional_kwargs={}),
     AIMessage(content='whats up?', additional_kwargs={})]
```

메모리를 체인, Agent에서 활용할 수 있는 다양한 방법을 제공한다. 

<img src="/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619184548995.png" alt="image-20230619184548995" style="zoom:30%;" />

# 5. Agents

일부 애플리케이션은 유저 입력에 기반하여 LLM 및 기타 도구의 유연한 체인 호출을 필요로 한다. Agent 인터페이스는 이러한 애플리케이션에 대한 유연성을 제공한다. Agent는 도구 모음에 액세스하며, 사용자 입력에 따라 어떤 도구를 사용할지 결정한다. Agent는 여러 도구를 활용할 수 있고, 하나의 도구의 아웃풋이 다른 도구의 인풋으로도 활용될 수 있다. 

Agent에는 두 가지 타입이 존재한다.

- Action agents: 각 단계에서, 이전 액션의 출력을 사용하여 다음 액션을 결정한다. 
- Plan-and-execute agents: 처음부터 모든 액션 시퀀스를 결정한 다음, 계획을 업데이트하지 않고 모두 실행한다. 

Plan-and-execute agent가 Action agent를 사용하여 계획을 실행하도록 하는 것이 좋은 방법이 될 수 있다. 

Agent type의 전체 list는 [agent types](https://python.langchain.com/docs/modules/agents/agent_types/)를 참조하면 된다. Agent에 관련된 추가적인 추상화 요소는 다음과 같다. 

- Tools: 에이전트가 수행할 수 있는 작업들. 에이전트에게 어떤 도구를 제공하는지는 에이전트가 어떤 작업을 수행하기를 원하는지에 크게 의존하게 된다. 
- Toolkits: 특정 유즈 케이스에 함께 사용될 수 있는 도구 모음의 래퍼. (예를 들어 에이전트가 SQL 데이터베이스와 상호 작용하려면 쿼리 실행 도구와 테이블 검사 도구가 필요할 것)

## Action agents

high-level에서 action agent는, 

1. 유저 인풋을 입력으로 받는다. 
2. 어떤 툴을 사용할지, 툴의 인풋을 결정한다. 
3. 툴을 호출하고 아웃풋을 기록한다. (also known as an "observation")
4. 툴 히스토리, 인풋, observation을 이용해 다음 스텝을 결정한다. 
5. 유저에게 직접적으로 응답할 수 있음을 결정할 때까지 3-4번 반복한다. 

Action agent들은 agent executor로 감싸져있고, 이것들이 agent를 호출하고, 액션과 액션 인풋을 받고, 해당 액션이 참조하는 도구를 호출하여 생성된 입력과 함께 도구를 호출하고, 도구의 출력을 가져온 다으메, 그 모든 정보를 에이전트에게 전달하여 다음에 취해야 할 액션을 가져온다. 

에이전트의 구성요소는 일반적으로 다음과 같다. 

- Prompt template : 유저 인풋과 이전 단계들을 받아서 프롬프트를 생성해서 언어모델로 전달한다.
- Language model :사용자 입력 및 액션 히스토리와 함께 프롬프트를 전달하고 다음에 할 작업을 결정한다.
- Output parser : 언어 모델의 출력을 가져와 다음 액션 또는 최종 답변으로 파싱한다. 

## Get Started

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# language model to use to control the agent
llm = OpenAI(temperature=0)

# let's load some tools to use. Note that `llm-math`tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

```

Agent한테 도구와 LLM을 제공해서, 질문에 대한 답을 하기 위해 어떤 도구를 사용할 것인지 결정하게 한다. 

```python
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```

![image-20230619203428933](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230619203428933.png)

## Conversational

다음은 대화에 최적화된 agent를 어떻게 활용하는지에 대한 데모이다. 최적의 응답을 찾기 위해 툴을 사용하는 데에 최적화된 agent들은 대화 셋팅에서는 이상적이지 않을 수 있다. 

`conversational-react-description`이라는 특정 타입의 에이전트를 활용할 수 있고, memory 컴포넌트와 함께 사용된다. 

```python
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

llm=OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run(input="hi, i am bob")
```

![image-20230620090741416](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230620090741416.png)

![image-20230620090753622](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230620090753622.png)

![image-20230620090807944](/Users/amy.yum/Library/Application Support/typora-user-images/image-20230620090807944.png)
