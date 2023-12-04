---

---

[다음](https://gbaccetta.medium.com/flutter-scalable-app-guide-part-1-architecture-8f60a2bbfe04) Medium Article 내용을 정리한 글입니다. 

# Introduction

Goal: to build an architecture that adheres to the following principles:

* **Domain-driven architecture,** with the domain layer as the middle layer that describes our models, the use cases, and the possible failures we need to handle.
* **Separation of concerns between the data layer and the UI layer,** with no dependencies between them.
* Adherence to the **Single Responsibility Principle (SRP).**
* Test coverage of 100%.

## The architecture infographic

![img](https://miro.medium.com/v2/resize:fit:2000/1*b5DZevkwmfHi5XmC-YBhiQ.png)

## DOMAIN

In the domain layer, we define the **data models** and **business use cases** of the application. 

We can think of the domain layer as the backbone that connects the **different parts of the application and defines how data is represented (MODELS)** and **handled (USES CASES)** by the different components in the DATA and UI layers

The domain layer will contain:

* **MODELS**
  * each model should represent a real-world concept or object, and contain an series of methods and properties(helpers) that are used either by the data layer or the UI layer.

* **FAILURES**
  * 
