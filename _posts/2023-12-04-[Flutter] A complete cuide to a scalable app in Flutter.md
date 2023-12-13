---
title: A complete guide to a scalable app in Flutter (Medium Article)
date: 2023-12-4
tags: 
    - flutter
    - Clean_Architecture
categories: 
    - flutter
toc: true
toc_sticky: true
toc_label: "페이지 목차"
---

[다음](https://gbaccetta.medium.com/flutter-scalable-app-guide-part-1-architecture-8f60a2bbfe04) Medium Article 내용을 정리한 글입니다. 

## Introduction

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
  * They are an important part of the domain layer. They help us identify known issues and customize the app’s behavior when those issues occur and minimizing the risk of having unhandled exceptions.
* **USE-CASES** -- *is it different from View Model?*
  * They are used to define the operations that the data layer should expose to the UI layer. They act as an interface between the two layers.
  * One-to-one relationsip between model and UseCase interface.

## DATA LAYER

[GO TO THE IMG](#The-architecture-infographic)

The data layer is responsible for accessing and manipulating data in the application, and should not have any knowledge of how that data is being used by other parts of the application. 

Two main types of components : modules and interactors.

* MODULES
  * They are contained within the data layer and are used exclusively by intectors. 
  *  They usually handle communication with **external components such as databases or API clients**, or they can act as background services as well as, when using Flutter, as plugins that interact with native platforms.

* INTERACTORS
  * Interactors are the concrete implementation of our use case contracts, making them the cornerstone of the data layer.
  * Interactors are responsible for handling business logic and data manipulation, and provide an interface for communication between the data layer sources and the rest of the application.

## UI LAYER

The UI layer is responsible for presenting the data and user interface of the application to the user. **It should only depend on the models and use cases described in the domain layer**, allowing changes to the domain and data layers without impacting the UI.

Inside the UI layer itself, we will achieve further decoupling by using a version of the **Model-View-ViewModel (MVVM) pattern** specifically adapted for Flutter.

* **VIEWS**
  * The Views are responsible for displaying the data and receiving user input.
  * They should not contain any business logic or data access code but delegate those responsibilities to the ViewModels. In flutter the views will contain everything related to widgets and context-related logic. 
  * By decouping the presentation from the logic itself, we can modify the UI without affecting the underlying logic and vice versa.
* **VIEW MODELS**
  * The ViewModels act as an intermediary between the View and the rest of the application. 
  * They make use of our use cases desribed in the domain to provide data and update the STATE upon which the views build themselves.
  * They will also fire events when a piece of logic is complete
  * view models handle all scenarios that we expect and correctly process all expected user input events (*i.e. tapOnSubmitFormButton()*)
* **STATES**
  * States hold a snapshot of all the variables needed to build a VIEW. 
  * They are updated by the VIEW MODELS and read by the Views. 
  * in Flutter, we will use ChangeNotifier to make the views aware when a change in the current STATE occurs. 
* **WIDGETS**
  * we can often create reusable components to be reused in different views.
* **ROUTER and ROUTES**
  * These components are responsible for managing the navigation and flow between different screens and views in the application. 

## DEPENDENCIES

Few key rules:

1. Arrows are unidirectional
2. When the line represents a solid trait, information can flow in both directions following a request/response pattern.
3. When the line is dotted, information can only flow in one direction using an events/subscription pattern.



## TESTING

Two kinds of tests:

* **UNIT TESTING**
  * Unit testing should be used to test individual components of the application in isolation. 
  * We can use mocking frameworks to simulate the behavior of external dependencies such as a database or API client.
* **FUNCTIONAL TESTING**
  * Functional testing or integration testing should be used to test the interactions between different components of the application.
  * Functional testing can help us identify issues that may arise due to the interactions between different components. 

Each of our architecture layers will have a corresponding testing strategy. The goal is to test the behavior of each component independently of the rest of the application, using mocked versions of the components that the tested component depends on as well as mocked data to simulate different scenarios.

- Modules: they can sometimes be partially or completely excluded from testing and coverage as they interact with external components. We will try to cover as much logic as we can with unit tests but in some cases will only create mock versions of our modules to simulate different scenarios.
- Interactors: since, as we said these are the cornerstone of the data layer, each interactor will have a corresponding testing file with unit tests covering every public method exposed by the interactor. We will use the mocked version of our modules and the mocked data.
- View Models: Similar to interactors, view models are the cornerstone of the UI layer. Hence, we will cover every method exposed in the view model contract with unit tests. We will use mocked versions of our interactors.
- Widgets: they are our reusable components for the views. In order to test them only once, unit tests for the different states will be written in a separate file for every widget.
- Views: testing them will allow us to perform Functional testing. This is the only place where our principle of independence is obviously broken. With functional testing, we test the actual flow of our app, including all dependencies. Hence, we will use real view models and interactors instead of their mocked versions. On the other hand, we will still need to use the mocked version of our modules as the external environment can rarely be simulated correctly during testing.

## The app folder structure and naming strategy

 a clean and well-organized structure that accurately reflects the chosen architecture will greatly improve code readability, maintainability, and scalability. 

### The app folder structure

> Exmple
>
> ![img](https://miro.medium.com/v2/resize:fit:544/1*ENX7pZXHnbflEiN7Yj0mtw.png)

### The testing folder structure

> Example
>
> ![img](https://miro.medium.com/v2/resize:fit:836/1*87-S8WpAPM8oAUNWFz_1dg.png)

