# -*- coding: utf-8 -*-
# last update: Sep.5 24
# author: Sean

import torch
import asyncio
import numpy as np
import sounddevice as sd
import warnings
warnings.filterwarnings("ignore")
from aiohttp import web
from rich.console import Console

from workflow_pipeline import ChatbotEventPipeline
from task_schedule import business_workflow
from utils import int2float

console = Console()
# init args for audio recording
task_id = "sean_mbp"


async def func_handle_request(request):
    data = await request.json()
    task_type = data['task_type']
    task_id = data['task_id']
    task_data = data['task_data']
    if task_type and task_data:
        pipeline.put_data((task_type, (task_id, task_data)))
        return web.json_response({'status': 'success'})
    else:
        return web.json_response({'status': 'failed'})

async def init_app():
    app = web.Application()
    app.router.add_post('/add_task', func_handle_request)
    return app

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

    def callback(indata, frames, time, status):
        if status:
            console.print("[blue]Audio indata status: ", status)
        func_vad(pipeline, task_id, indata.copy())

    with sd.InputStream(
        samplerate=16000,
        dtype='int16',
        channels=1,
        callback=callback,
        blocksize=512):

        console.print(f"[blue]Start audio recording...")
        while True:
            await asyncio.sleep(0.01)

async def main_by_request(pipeline):    
    # create pipeline task
    pipeline_task = asyncio.create_task(pipeline.execute())

    # start web server
    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()

    # wait for all data to be processed
    await pipeline.queue.join()

    # wait for pipeline task to finish
    await pipeline_task

async def main_by_audio(pipeline):

    # audio recording
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
        你是一个聊天机器人，你的任务是与用户进行自然对话。请尽量让对话自然流畅，避免出现不连贯的情况。你可以随时结束对话，但请不要在中途中断对话。
    """

    pipeline = ChatbotEventPipeline(service_name="deepseek",
                                system_prompt=system_prompt,
                                workflow=business_workflow)

    # asyncio.run(main_by_request(pipeline=pipeline))
    asyncio.run(main_by_audio(pipeline=pipeline))