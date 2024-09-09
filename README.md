# Lite Speech2Speech Pipeline

A tool that allows users to quickly customize LLM chatbot workflow pipelines, like Text-to-Text, Text-to-Speech or Speech-to-Speech

## Overview

This project is an asynchronous LLM chatbot task workflow management tool, where each function is an independent module, and the data input and output are circulated in the data flow pipeline.

![pipeline structure](./docs/img/pipeline%20structure.png)

## Module

The entire workflow is executed through three modules, task_schedule, func_tools, workflow_pipeline, and main is the entry point of the program.

- task_schedule: responsible for task scheduling
- func_tools: responsible for the specific implementation of tasks
- workflow_pipeline: responsible for the operation of the task pipeline and environment initialization
- main: program entry, defines the input and output of workflow_pipeline (API or Audio)

![pipeline structure](./docs/img/pipeline%20function%20implementation.png)

## Instance

This project implements two instances, Speech-to-Speech and Text-to-Speech.

> VAD, TTS, and STT refer to the implementation code in HuggingFace's [speech-to-speech](https://github.com/huggingface/speech-to-speech.git).

- VAD: Silero VAD (adding additionally ability to interrupt audio in real time)
- STT: Distil-Whisper
- LLM: DeepSeek-chatbot
- TTS: MeloTTS
- Audio input/output: sounddevice

![pipeline instance](./docs/img/pipeline%20instance.png)

## Usage

```shell
# clone the repo
git clone https://github.com/sxinyuhoo/lite-speech2speech-pipeline.git
cd lite-speech2speech-pipeline

# install the dependencies
pip install -r requirements.txt

# configure the LLM API-KEY in the `config.ini` file

# run the main.py
python text2speech/main.py 
# or 
python speech2speech/main.py

```

While using text2speech, you can send a request to chatbot like this:

execute `python text2speech/main.py`, and then send a request through `curl -X POST http://localhost:8080/add_task -H "Content-Type: application/json" -d '{"task_type": "chatbot", "task_data": "hello", "task_id": "sean"}'`.

## Citation

- VAD

```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```

- STT

```bibtex
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling},
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

- TTS

```bibtex
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

- speech-to-speech

```bibtex
@software{speech-to-speech,
  author = {Hugging Face},
  title = {Speech To Speech: an effort for an open-sourced and modular GPT4-o},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/huggingface/speech-to-speech.git}
}
```
