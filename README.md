# Lite Speech2Speech Pipeline

A tool that allows users to quickly customize LLM chatbot workflow pipelines, like Text-to-Text, Text-to-Speech or Speech-to-Speech

## Design

![pipeline structure](./docs/img/pipeline%20structure.png)

本项目是基于异步实现的LLM chatbot任务工作流管理工具，每个function都是一个独立的模块，通过task_schedule决定模块各自之间的衔接关系，数据的输入输出在数据流管道中进行流转。

## Module

![pipeline structure](./docs/img/pipeline%20function%20implementation.png)

通过三个模块实现整个工作流程的管理，分别是task_schedule、tools、workflow_pipeline。
task_schedule负责任务的调度，tools负责任务的管理，workflow_pipeline负责任务管道的流转。

## Instance

本项目实现了两个实例，分别是Text-to-Text和Speech-to-Speech。

![pipeline instance](./docs/img/pipeline%20instance.png)

## Usage
