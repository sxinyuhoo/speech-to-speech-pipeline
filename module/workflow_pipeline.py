# -*- coding: utf-8 -*-
# last update: Sep.2 24
# author: Sean

import torch
import asyncio
import inspect
from utils import gen_client

from vad_iterator import VADIterator
from lightning_whisper_mlx import LightningWhisperMLX

# create pipeline class
class ChatbotEventPipeline:

    def __init__(self,
                service_name="deepseek",
                system_prompt="",
                workflow={},
                stt_model_name="distil-large-v3"
                ):
        self.queue = asyncio.Queue()
        self.state = 'RUNNING'
        self.system_prompt = system_prompt
        self.business_workflow = workflow
        self.session_data = {}
        self.client, self.model = gen_client(service_name=service_name, if_async=True)

        # VAD
        self.vad_sample_rate = 16000
        self.vad_min_silence_ms = 1000
        self.vad_min_speech_ms = 1000
        self.vad_max_speech_ms = float('inf')
        self.vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=0.3,
            sampling_rate=self.vad_sample_rate,
            min_silence_duration_ms=self.vad_min_silence_ms,
            speech_pad_ms=30,
        )
        # STT 
        self.stt_model = LightningWhisperMLX(model=stt_model_name, batch_size=6, quant=None)

    def add(self, process_name, func):
        self.business_workflow[process_name] = func
    
    def pause(self, process_name):
        pass
    
    def resume(self, process_name):
        pass

    def put_data(self, data):
        self.queue.put_nowait(data)

    async def execute(self):
        while self.state:
            nxt_func, args = await self.queue.get()
            usr_id, data = args[0], args[1]

            # set the execute to stop
            if nxt_func == 'STOP':
                self.state = False

            elif nxt_func == 'None':
                pass

            elif nxt_func in self.business_workflow:
                func = self.business_workflow[nxt_func]

                # check if the function is async generator function
                if inspect.isasyncgenfunction(func):
                    async for _nxt, _args in func(self, usr_id, data):
                        _usr_id, _data = _args[0], _args[1]
                        # print("workflow_pipeline agf", _nxt, _usr_id, _data)
                        await self.put_data((_nxt, (_usr_id, _data)))
                else:
                    _nxt, _args = await func(self, usr_id, data)
                    _usr_id, _data = _args[0], _args[1]
                    # print("workflow_pipeline", _nxt, _usr_id, _data)
                    await self.put_data((_nxt, (_usr_id, _data)))

            self.queue.task_done()



