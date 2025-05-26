import argparse

import datasets
import json
import pdb
import time
import os
import requests
import torch
import gc
import psutil
import argparse

from config import *
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from openai import OpenAI
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

current_file_path = os.path.abspath(__file__)
base_directory = os.path.dirname(current_file_path)


def get_data(only_en:bool = False):
    if only_en: 
        return datasets.load_dataset("WYRipple/S1-Bench", split='train').filter(lambda x : x["language"] == 'en')
    else: 
        return datasets.load_dataset("WYRipple/S1-Bench", split='train')


# def append_to_jsonl(file_path, data_dict):
#     json_str = json.dumps(data_dict, ensure_ascii=False) + '\n'
#     with open(file_path, 'a', encoding='utf-8') as f:
#         f.write(json_str)


def init_main(model_name, k, is_greedy, only_en):
    local_model_list = {
        "DS-R1-1.5B": "local-model-path",
        "DS-R1-7B": "local-model-path",
        "DS-R1-8B": "local-model-path",
        "DS-R1-14B": "local-model-path",
        "DS-R1-32B": "local-model-path",
        "DS-R1-70B": "local-model-path",
        "QwQ-32B": "local-model-path",
        "L-R1-7B-DS": "local-model-path",
        "L-R1-14B-DS": "local-model-path",
        "L-R1-32B-DS": "local-model-path",
        "L-R1-32B": "local-model-path",
        "s1.1-7B": "simplescaling/s1.1-7B",
        "s1.1-14B": "simplescaling/s1.1-14B",
        "s1.1-32B": "simplescaling/s1.1-32B",
        "EXAONE-2.4B": "local-model-path",
        "EXAONE-7.8B": "local-model-path",
        "EXAONE-32B": "local-model-path",
        "Nemotron-8B": "local-model-path",
        "Nemotron-49B": "local-model-path",
        "Sky-T1-32B": "local-model-path",
        # "datumo": "/home/elicer/s1/ckpts/20250512_024851-gemini-enforced-connector-failed-norepeat-error-v2-e5-23000",
        "datumo": "/home/elicer/s1/ckpts/20250515_035228-enforced-connector-block4000",
        "datumo-incorrect": "/home/elicer/s1/ckpts/20250515_154904-enforced-connector-without-correct-block4000-e5",
        "datumo-correct": "/home/elicer/s1/ckpts/20250518_054331-enforced-connector-without-incorrect-block4000-e5",
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "LIMO": "/home/elicer/s1/ckpts/20250515_001008-limo-block13000"
    }

    """ data prepare """
    
    data_list = get_data(only_en)
    
    print(f"读取eval数据 {len(data_list)} 条")

    """ begin load model """
    tokenizer = AutoTokenizer.from_pretrained(local_model_list[model_name])
    model = LLM(model=local_model_list[model_name], 
                trust_remote_code=True, 
                gpu_memory_utilization=0.95,
                tensor_parallel_size=MODEL_CONFIG[model_name]['n_gpu'],
                max_num_seqs=64,
                )

    if is_greedy:
        sampling_params = SamplingParams(temperature=0.0, 
                                        stop_token_ids=[tokenizer.eos_token_id],
                                        max_tokens=10000,
                                        skip_special_tokens=False,
                                        )
    else:
        sampling_params = SamplingParams(temperature=0.6, 
                                        top_p=0.95, 
                                        stop_token_ids=[tokenizer.eos_token_id],
                                        max_tokens=10000,
                                        skip_special_tokens=False,
                                        )
    
    print(f"LRM is [ {model_name} ]")
    for i in range(k):
        if is_greedy:
            output_path = base_directory + f"/LRM_local_vllm_temp_0/{model_name}/LRM_response_{model_name}_{str(i)}.json"
        else:
            output_path = base_directory + f"/LRM_local_vllm/{model_name}/LRM_response_{model_name}_{str(i)}.json"
        prompts = []

        # data_list = data_list[:8]
        
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        for sample in data_list:
            if MODEL_CONFIG[model_name]['system_prompt'] != False:
                msg = [{"role": "system", "content": MODEL_CONFIG[model_name]['system_prompt']}, {"role": "user", "content": sample["question"]}]
            else:
                msg = [{"role": "user", "content": sample["question"]}]
            
            text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            if model_name in ["L-R1-7B", "L-R1-14B", "L-R1-32B", "Sky-T1-32B"]:
                text += "<think>\n"
            elif model_name in ["s1.1-7B", "s1.1-14B", "s1.1-32B", "Trilogix-7B", 'datumo']:
                text += "<|im_start|>think"
            prompts.append(text)

        responses = model.generate(prompts, sampling_params=sampling_params)

        save_samples = []
        for sample, response in zip(data_list, responses):
            sample["test_model"] = model_name
            sample["model_response"] = response.outputs[0].text
            save_samples.append(sample)
        # pdb.set_trace()
        
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(save_samples, json_file, ensure_ascii=False, indent=4)
        print(f"Save the results to {output_path}")
        # pdb.set_trace()



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--greedy", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--only_en", type=lambda x: x.lower() == "true", default=False)
    args = parser.parse_args()

    init_main(args.model, k=args.k, is_greedy=args.greedy, only_en=args.only_en)