import pdb
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import re
from pprint import pprint
current_file_path = os.path.abspath(__file__)
base_directory = os.path.dirname(current_file_path)
import argparse

def read_jsonl_to_list(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                result.append(data)
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--greedy", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--s1bench_len", type=int, required=True)
    args = parser.parse_args()

    model_list = [args.model]
    is_greedy = args.greedy

    model_results = {} 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    s1bench_len = args.s1bench_len

    for model_name in model_list:
        model_stats = {
            "pass_acc_loose": 0,
            "pass_acc_strict": 0,
            "total_acc_loose": 0,
            "total_acc_strict": 0,
            "success_rate_loose": 0,
            "success_rate_strict": 0,
            "avg_tokens_strict": 0,
            "avg_tokens_loose": 0,
        }

        data_list_5 = []
        eval_path = "LRM_acc_eval_temp_0" if is_greedy else "LRM_acc_eval"

        n_files = 1 if is_greedy else 5
        for i in range(n_files):
            file_path = os.path.join(base_dir, f"{eval_path}/{model_name}/LRM_response_eval_{model_name}_{i}.json")
            data_list = read_jsonl_to_list(file_path)
            data_list_5.extend(data_list)

        loose_pass = strict_pass = loose_total = strict_total = 0
        total_think_token_strict = total_answer_token_strict = 0
        total_think_token_loose = total_answer_token_loose = 0
        success_num_loose = success_num_strict = 0

        for this_id in range(1, s1bench_len + 1):
            total_flag_easy = total_flag_hard = 0
            for sample in data_list_5:
                if this_id == sample["ID"]:
                    if sample["think_success"] in [100, 101]:  # strict
                        total_think_token_strict += sample["thinking_part_tokens"]
                        total_answer_token_strict += sample["answering_part_tokens"]
                        success_num_strict += 1
                        if sample["eval_result"] == 1:
                            total_flag_hard += 1
                            strict_pass += 1
                    if sample["think_success"] not in [300, 301]:  # loose
                        total_think_token_loose += sample["thinking_part_tokens"]
                        total_answer_token_loose += sample["answering_part_tokens"]
                        success_num_loose += 1
                        if sample["eval_result"] == 1:
                            total_flag_easy += 1
                            loose_pass += 1

            if total_flag_easy == n_files:
                loose_total += 1
            if total_flag_hard == n_files:
                strict_total += 1

        model_stats["pass_acc_loose"] = loose_pass / (n_files * s1bench_len)
        model_stats["pass_acc_strict"] = strict_pass / (n_files * s1bench_len)
        model_stats["total_acc_loose"] = loose_total / s1bench_len
        model_stats["total_acc_strict"] = strict_total / s1bench_len
        model_stats["success_rate_loose"] = success_num_loose / (n_files * s1bench_len)
        model_stats["success_rate_strict"] = success_num_strict / (n_files * s1bench_len)

        if success_num_strict == 0:
            model_stats['avg_tokens_strict'] = 0
            print("STRICT으로 성공한 데이터가 없습니다.")
        else:
            model_stats["avg_tokens_strict"] = (total_think_token_strict + total_answer_token_strict) / success_num_strict
        model_stats["avg_tokens_loose"] = (total_think_token_loose + total_answer_token_loose) / success_num_loose

        model_results[model_name] = model_stats

    pprint(model_results, indent=4)
