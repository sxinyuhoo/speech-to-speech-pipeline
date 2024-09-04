# -*- coding: utf-8 -*-
# last update: Sep.5 24
# author: Sean

from func_tools import func_stt, func_chatbot, func_tts, func_audio_output

business_workflow = {
    'stt': func_stt,
    'chatbot': func_chatbot,
    'tts': func_tts,
    'output': func_audio_output,
}
