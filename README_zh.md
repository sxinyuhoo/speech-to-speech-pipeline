# Lite Speech2Speech Pipeline

A tool that allows users to quickly customize LLM chatbot workflow pipelines, like Text-to-Text, Text-to-Speech or Speech-to-Speech

## Design

![pipeline structure](./docs/img/pipeline%20structure.png)

本项目是基于异步实现的LLM chatbot任务工作流管理工具，每个function都是一个独立的模块，通过task_schedule决定模块各自之间的衔接关系，数据的输入输出在数据流管道中进行流转。

## Structure

![pipeline structure](./docs/img/pipeline%20function%20implementation.png)

该实现思路与huggingface/speech to speech项目类似（那项目使用多线程和多队列实现），但相对实现了一个pipeline，通过
