import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HF_HOME'] = '/mnt/jaewoo4tb/cache/'

from PIL import Image
import requests
import copy
import torch
import time
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
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
        pretrained = "lmms-lab/llama3-llava-next-8b"
        model_name = "llava_llama3"
        print("start")
        # pretrained = "lmms-lab/llava-next-72b"
        # model_name = "llava_llama3"
        self.device = "cuda"
        device_map = "auto"
        self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) # Add any other thing you want to pass in llava_model_args
        self.model.eval()
        self.model.tie_weights()

    @logging_time
    def get_caption(self, image, question):
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
        # conv_template = "qwen_1_5" # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]        
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=50,
            pad_token_id=self.tokenizer.eos_token_id
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs
