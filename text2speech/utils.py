# -*- coding: utf-8 -*-
# last update: Sep.2 24
# author: Sean

import os
import configparser
import numpy as np
from openai import OpenAI, AsyncOpenAI

def int2float(sound):
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def gen_client(service_name, if_async=False):

    config = configparser.ConfigParser()

    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.ini")
    config.read(config_path)

    deepseek_api_key = config['API_KEY']['deepseek_api_key']
    chatglm_api_key = config['API_KEY']['chatglm_api_key']

    if service_name == "deepseek":
        api_key = deepseek_api_key
        base_url = "https://api.deepseek.com/v1"
        model = "deepseek-chat"
    elif service_name == "chatglm":
        api_key = chatglm_api_key
        base_url = "https://open.bigmodel.cn/api/paas/v4/"
        model = "glm-4"
    else:
        raise ValueError("Invalid service name")
    
    if if_async:
        client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
    else:
        client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
    return client, model

if __name__ == '__main__':
    gen_client("deepseek", if_async=False)