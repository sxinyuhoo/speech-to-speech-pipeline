# -*- coding: utf-8 -*-
# last update: Sep.6 24
# author: Sean

import torch
import asyncio
import inspect
from rich.console import Console
from melo.api import TTS

from utils import gen_client

from vad_iterator import VADIterator
from lightning_whisper_mlx import LightningWhisperMLX

console = Console()

WHISPER_LANGUAGE_TO_MELO_LANGUAGE = {
    "en": "EN_NEWEST",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}

WHISPER_LANGUAGE_TO_MELO_SPEAKER = {
    "en": "EN-Newest",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}

# create pipeline class
class ChatbotEventPipeline:

    def __init__(self,
                service_name="deepseek",
                system_prompt="",
                workflow={},
                stt_model_name="distil-large-v3"
                ):

        self.queue = asyncio.Queue() # pipeline queue
        self.audio_output_queue = asyncio.Queue() # audio output queue
        self.state = 'RUNNING'
        self.system_prompt = system_prompt
        self.business_workflow = workflow
        self.session_data = {}
        self.client, self.model = gen_client(service_name=service_name, if_async=True)
        
        self.device = "cpu"

        # VAD
        console.print("[blue]ChatbotEventPipeline: initializing VAD...")
        self.vad_sample_rate = 16000
        self.vad_min_silence_ms = 1000
        self.vad_min_speech_ms = 1000
        self.vad_max_speech_ms = float('inf')
        self.vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=0.5,
            sampling_rate=self.vad_sample_rate,
            min_silence_duration_ms=self.vad_min_silence_ms,
            speech_pad_ms=30,
        )
        # STT 
        console.print("[blue]ChatbotEventPipeline: initializing STT...")
        self.stt_model = LightningWhisperMLX(model=stt_model_name, batch_size=6, quant=None)

        # TTS
        console.print("[blue]ChatbotEventPipeline: initializing TTS...")
        self.tts_language = 'en'
        self.tts_speaker = 'en'
        self.tts_model = TTS(
            language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[self.tts_language], device=self.device
        )
        self.tts_speaker_id = self.tts_model.hps.data.spk2id[
            WHISPER_LANGUAGE_TO_MELO_SPEAKER[self.tts_speaker]
        ]
        self.blocksize = 512

        console.print("[blue]ChatbotEventPipeline: initialization complete")

    def add(self, process_name, func):
        self.business_workflow[process_name] = func
    
    def pause(self, process_name):
        pass
    
    def resume(self, process_name):
        pass

    def put_data(self, data):
        self.queue.put_nowait(data)

    async def handle_async_gen_func(self, func, usr_id, data):
        async for _nxt_task, _args in func(self, usr_id, data):
            _usr_id, _data = _args[0], _args[1]
            self.put_data((_nxt_task, (_usr_id, _data)))

    async def handle_async_func(self, func, usr_id, data):
        _nxt_task, _args = await func(self, usr_id, data)
        _usr_id, _data = _args[0], _args[1]
        self.put_data((_nxt_task, (_usr_id, _data)))

    async def execute(self):
        while self.state:
            task, args = await self.queue.get()
            usr_id, data = args[0], args[1]

            # set the execute to stop
            if task == 'STOP':
                self.state = False

            elif task == 'None':
                pass

            elif task in self.business_workflow:
                func = self.business_workflow[task]

                # check if the function is async generator function
                if inspect.isasyncgenfunction(func):
                    asyncio.create_task(self.handle_async_gen_func(func, usr_id, data))
                else:
                    asyncio.create_task(self.handle_async_func(func, usr_id, data))

            self.queue.task_done()



