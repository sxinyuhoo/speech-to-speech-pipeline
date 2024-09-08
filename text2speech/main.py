# -*- coding: utf-8 -*-
# last update: Sep.8 24
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

async def func_handle_audio_output(pipeline):

    def callback(outdata, frames, time, status):
        try:
            audio_data = pipeline.audio_output_queue.get_nowait()
            outdata[:] = audio_data.reshape(outdata.shape)
        except Exception as e:
            outdata[:] = 0 * outdata

    with sd.OutputStream(
        samplerate=16000,
        dtype='int16',
        channels=1,
        callback=callback,
        blocksize=512):

        while True:
            await asyncio.sleep(0.01)

async def main_by_request(pipeline):    
    # create pipeline task
    pipeline_task = asyncio.create_task(pipeline.execute())

    # audio task
    audio_task = asyncio.create_task(func_handle_audio_output(pipeline))

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
    await audio_task

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
    
    asyncio.run(main_by_request(pipeline=pipeline))