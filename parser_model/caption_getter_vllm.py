import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HF_HOME'] = '/mnt/jaewoo4tb/cache/'
from io import BytesIO

import requests
from PIL import Image
import numpy as np
import time
from vllm import LLM, SamplingParams

# import copy
import torch
# import time
def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn


class CaptionModel:
    def __init__(self):
        self.llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf", max_model_len=4096)

    @logging_time
    def get_caption(self, image, question):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        with torch.no_grad():
            sampling_params = SamplingParams(temperature=0.8,
                                    top_p=0.97,
                                    max_tokens=40)
            outputs = self.llm.generate(
                    {
                        "prompt": question,
                        "multi_modal_data": {
                            "image": image
                        }
                    },
                    sampling_params=sampling_params)
            generated_text = ""
            for o in outputs:
                generated_text += o.outputs[0].text
            return generated_text