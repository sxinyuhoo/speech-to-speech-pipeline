# -*- coding: utf-8 -*-
# last update: Sep.2 24
# author: Sean

import torch

### Tools Inner Functions ###

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
        # output += i.choices[0].delta.content
        # yield i.choices[0].delta.content
        
    output_fmt = message_fmt + [{'role': 'assistant', 'content': output}]

    # update session data
    await update_session_data(pipeline=pipeline, session_id=session_id, new_conversation=output_fmt)

    # return output

### Tools for workflow pipeline ###

async def func_stt(pipeline, user_id, audio_data):

    print("inferring whisper...")
    inferred_text = pipeline.stt_model.transcribe(audio_data, language='zh')["text"].strip()
    torch.mps.empty_cache()

    print(f"[YELLOW]STT USER | {user_id}: {inferred_text}")
    return ('chatbot', (user_id, inferred_text))

async def func_chatbot(pipeline, user_id, msg):
    # print("chatbot | user_id:", user_id, "| message:", msg)
    
    client, model, system_prompt = pipeline.client, pipeline.model, pipeline.system_prompt

    # output_msg = ""
    # output_msg = await async_chat(pipeline=pipeline, session_id=user_id, message=msg, system_prompt=system_prompt, client=client, model=model)
    async for _output in async_chat(pipeline=pipeline, session_id=user_id, message=msg, system_prompt=system_prompt, client=client, model=model):
        # print(f"[green] CHATBOT: {_output}")
        yield ('output', (user_id, _output))

    # return ('output', user_id, output_msg) # define next function to call

async def func_output(pipeline, user_id, msg):
    # print("stream output | user_id:", user_id, "| message:", msg)
    print(f"[GREEN] CHATBOT | {user_id}: {msg}")
    return ('None', (None, None))



