# -*- coding: utf-8 -*-
# last update: Sep.9 24
# author: Sean

import torch
import asyncio
import numpy as np
import sounddevice as sd

from aiohttp import web
from rich.console import Console

from workflow_pipeline import ChatbotEventPipeline
from task_schedule import business_workflow
from utils import int2float

console = Console()
# init args for audio recording
task_id = "default"

def func_vad(pipeline,
                user_id,
                audio_chunk):

    audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
    audio_float32 = int2float(audio_int16)
    vad_output = pipeline.vad_iterator(torch.from_numpy(audio_float32))

    if vad_output is not None and len(vad_output) > 0:
        console.print("[blue]VAD: end of speech detected")

        array = torch.cat(vad_output).cpu().numpy()
        duration_ms = len(array) / pipeline.vad_sample_rate * 1000

        if duration_ms < pipeline.vad_min_speech_ms or duration_ms > pipeline.vad_max_speech_ms:
            console.print(f"[blue]VAD: audio input of duration: {len(array) / pipeline.vad_sample_rate}s, skipping")
        else:
            console.print("[blue]VAD: put data to pipeline")
            pipeline.put_data(('stt', (user_id, array)))

async def func_handle_audio(pipeline):

    def callback(indata, outdata, frames, time, status):
        if pipeline.audio_output_queue.empty():
            func_vad(pipeline, task_id, indata.copy())
            outdata[:] = 0 * outdata
        else:
            audio_data = pipeline.audio_output_queue.get_nowait()
            outdata[:] = audio_data.reshape(outdata.shape)

    with sd.Stream(
        samplerate=16000,
        dtype='int16',
        channels=1,
        callback=callback,
        blocksize=512):

        console.print(f"[blue]Start audio detect...")
        while True:
            await asyncio.sleep(0.01)

async def main_by_audio(pipeline):

    # audio task
    audio_task = asyncio.create_task(func_handle_audio(pipeline))

    # create pipeline task
    pipeline_task = asyncio.create_task(pipeline.execute())

    await audio_task

    # wait for all data to be processed
    await pipeline.queue.join()

    # wait for pipeline task to finish
    await pipeline_task

if __name__ == '__main__':


    system_prompt = """
        You are a chatbot, your task is to have a natural conversation with the user in a short response. 
        Please try to make the conversation flow naturally and avoid any incoherence. 
        You can end the conversation at any time, but please do not interrupt the conversation halfway.
    """
    # system_prompt_zh = """
    #     你是一个聊天机器人，你的任务是用简短的回答与用户进行自然对话。
    #     请尽量使对话流畅自然，避免任何不连贯。
    # """

    pipeline = ChatbotEventPipeline(service_name="deepseek",
                                system_prompt=system_prompt,
                                workflow=business_workflow)
    
    asyncio.run(main_by_audio(pipeline=pipeline))