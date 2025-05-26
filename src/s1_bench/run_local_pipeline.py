import os
import subprocess
import argparse

def run_step(step_name, script_path, args=""):
    print(f"\n🚀 Running [{step_name}]...")
    command = f"python3 {script_path} {args}"
    print(f"→ {command}")
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--greedy", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--only_en", type=lambda x: x.lower() == "true", default=False)
    args = parser.parse_args()

    model = args.model
    k = args.k
    is_greedy = args.greedy
    if args.only_en:
        s1bench_len = 220
    else:
        s1bench_len = 422

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    greedy_flag = "temp_0" if is_greedy else ""
    sampling_args = f"--model {model} --k {k} --greedy {is_greedy} --only_en {args.only_en}"

    run_step("Step 1: Generate responses", f"{base_dir}/get_LRM_vllm_response.py", sampling_args)
    run_step("Step 2: Split think/answer", f"{base_dir}/split_think_answer.py", f"--model {model} --greedy {is_greedy}")
    run_step("Step 3: Run evaluator", f"{base_dir}/get_LRM_eval.py", f"--model {model} --greedy {is_greedy}")
    run_step("Step 4: Compute accuracy", f"{base_dir}/get_acc_scores.py", f"--model {model} --greedy {is_greedy} --s1bench_len {s1bench_len}")

# python run_local_pipeline.py --model Trilogix-7B --k 5 --greedy False --only_en True
# python run_local_pipeline.py --model Trilogix-7B --k 1 --greedy True --only_en False
# python run_local_pipeline.py --model datumo --k 5 --greedy False --only_en False
# python run_local_pipeline.py --model datumo --k 1 --greedy True --only_en False
# python run_local_pipeline.py --model Qwen2.5-7B-Instruct --k 5 --greedy False --only_en False

# export CUDA_VISIBLE_DEVICES=0,2
# ================ GREEDY ================
# python run_local_pipeline.py --model Trilogix-7B --k 1 --greedy True --only_en False (Epoch 3)
# {   'Trilogix-7B': {   'avg_tokens_loose': 595.4569536423841,
#                        'avg_tokens_strict': 652.5314960629921,
#                        'pass_acc_loose': 0.6516587677725119,
#                        'pass_acc_strict': 0.5876777251184834,
#                        'success_rate_loose': 0.7156398104265402,
#                        'success_rate_strict': 0.6018957345971564,
#                        'total_acc_loose': 0.6516587677725119,
#                        'total_acc_strict': 0.5876777251184834}}

# python run_local_pipeline.py --model Trilogix-7B --k 1 --greedy True --only_en False (Epoch 2)
# {   'Trilogix-7B': {   'avg_tokens_loose': 346.6775147928994,
#                        'avg_tokens_strict': 366.94444444444446,
#                        'pass_acc_loose': 0.7440758293838863,
#                        'pass_acc_strict': 0.6706161137440758,
#                        'success_rate_loose': 0.8009478672985783,
#                        'success_rate_strict': 0.6824644549763034,
#                        'total_acc_loose': 0.7440758293838863,
#                        'total_acc_strict': 0.6706161137440758}}

# python run_local_pipeline.py --model datumo --k 1 --greedy True --only_en False
# {   'Trilogix-7B': {   'avg_tokens_loose': 309.13031914893617,
#                        'avg_tokens_strict': 313.24719101123594,
#                        'pass_acc_loose': 0.8530805687203792,
#                        'pass_acc_strict': 0.8246445497630331,
#                        'success_rate_loose': 0.8909952606635071,
#                        'success_rate_strict': 0.8436018957345972,
#                        'total_acc_loose': 0.8530805687203792,
#                        'total_acc_strict': 0.8246445497630331}}


# python run_local_pipeline.py --model s1.1-7B --k 1 --greedy True --only_en False
# {   's1.1-7B': {   'avg_tokens_loose': 595.1687979539641,
# 'avg_tokens_strict': 607.8232044198895,
# 'pass_acc_loose': 0.9241706161137441,
# 'pass_acc_strict': 0.8554502369668247,
# 'success_rate_loose': 0.9265402843601895,
# 'success_rate_strict': 0.8578199052132701,
# 'total_acc_loose': 0.9241706161137441,
# 'total_acc_strict': 0.8554502369668247}}