# -*- coding: utf-8 -*-
# last update: Sep.5 24
# author: Sean

import torch
import librosa
import numpy as np
import sounddevice as sd
from rich.console import Console

#############################
### Tools Inner Functions ###
#############################

console = Console()

# chat history management
# session_data structure: {session_id: [{conversation1}, {conversation2}, ...]}
# async update session data
async def update_session_data(pipeline, session_id, new_conversation, keep_alive_num=10):
    if session_id not in pipeline.session_data:
        pipeline.session_data[session_id] = new_conversation
    else:
        pipeline.session_data[session_id].extend(new_conversation)
    if len(pipeline.session_data[session_id]) > keep_alive_num:
        pipeline.session_data[session_id] = pipeline.session_data[session_id][-keep_alive_num:]

# async get session data
async def get_session_data(pipeline, session_id):
    if session_id in pipeline.session_data:
        return pipeline.session_data[session_id]
    else:
        return []

async def async_chat(pipeline, session_id, message, system_prompt, client, model):
    output = ""

    # get session data
    history = await get_session_data(pipeline, session_id)

    message_fmt = [{'role': 'user', 'content': message}]

    resp = await client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system_prompt}] + history + message_fmt,
        temperature=0.8,
        stream=True,
        )
    async for i in resp:
        cur_content = i.choices[0].delta.content
        output += cur_content
        yield cur_content
        
    output_fmt = message_fmt + [{'role': 'assistant', 'content': output}]

    # update session data
    await update_session_data(pipeline=pipeline, session_id=session_id, new_conversation=output_fmt)

###################################
### Tools for workflow pipeline ###
###################################

async def func_stt(pipeline, user_id, audio_data):

    console.print("[blue]inferring whisper...")
    inferred_text = pipeline.stt_model.transcribe(audio_data, language='zh')["text"].strip()
    torch.mps.empty_cache()

    console.print(f"[yellow]USER {user_id}: {inferred_text}")
    return ('chatbot', (user_id, inferred_text))

async def func_chatbot(pipeline, user_id, msg):
    console.print("[blue]chatbot processing...")
    
    client, model, system_prompt = pipeline.client, pipeline.model, pipeline.system_prompt

    curr_output = ""

    # output_msg = await async_chat(pipeline=pipeline, session_id=user_id, message=msg, system_prompt=system_prompt, client=client, model=model)
    async for _output in async_chat(pipeline=pipeline, session_id=user_id, message=msg, system_prompt=system_prompt, client=client, model=model):
        curr_output += _output
        if curr_output.endswith((".", "?", "!")):
            console.print(f"[green] CHATBOT : {curr_output}")
            yield ('tts', (user_id, curr_output))
            curr_output = ""

async def func_tts(pipeline, user_id, llm_sentence):
    console.print("[blue]tts processing...", llm_sentence)

    if pipeline.device == "mps":
        import time
        start = time.time()
        torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
        torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
        _ = (
            time.time() - start
        )  # Removing this line makes it fail more often. I'm looking into it.

    try:
        console.print("[blue]tts in the try...")
        audio_chunk = pipeline.tts_model.tts_to_file(
            llm_sentence, pipeline.tts_speaker_id, quiet=True
        )
        # console.print("[blue]tts tried the res: ", audio_chunk)
    except Exception as e:
        print("[blue]tts in the except: ", e)
        audio_chunk = np.array([])
    if len(audio_chunk) == 0:
        console.print("[blue]tts no audio chunk...")
        yield ('None', (None, None))
    
    audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
    audio_chunk = (audio_chunk * 32768).astype(np.int16)

    console.print("[blue]tts audio chunk yield processing...")
    for i in range(0, len(audio_chunk), pipeline.blocksize):
        yield ('output', 
            (user_id, 
            np.pad(
            audio_chunk[i : i + pipeline.blocksize],
            (0, pipeline.blocksize - len(audio_chunk[i : i + pipeline.blocksize])),
        )))

async def func_audio_output(pipeline, user_id, audio_chunk):
    
    # sd.play(audio_chunk, samplerate=16000)
    # sd.wait()

    return ('None', (None, None))



