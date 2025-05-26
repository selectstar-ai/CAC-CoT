import os
import argparse
import logging
import yaml
import concurrent.futures

from datasets import load_dataset, load_from_disk, DatasetDict
from typing import List, Tuple
from tqdm import tqdm
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

def setup_logging(log_dir: str = "logs/evaluate") -> str:
    """
    로그 디렉토리를 생성하고, 타임스탬프 기반 파일명으로 로깅을 초기화합니다.

    Returns:
        로그 파일 경로
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"analysis_{ts}.log")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    # HTTP 통신 로깅 수준 상향
    for lib in ["openai", "urllib3", "httpx"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.info("Logging initialized.")
    return log_file


def parse_args() -> argparse.Namespace:
    """
    CLI 인자를 파싱합니다.
    """
    p = argparse.ArgumentParser(description="Dataset analysis and optional GPT-based grading.")
    
    # -- DATA
    p.add_argument("--dataset_dir", required=True, help="HuggingFace 이름 또는 로컬 디스크 경로")
    p.add_argument("--type", choices=['hf', 'disk'], default='hf', help="데이터 로드 방식: hf (load_dataset) 또는 disk (load_from_disk)")


    # -- CONNECTOR
    p.add_argument("--connector_yaml", default="config/connector.yaml", help="커넥터 리스트 YAML 경로")

    # -- GRADING
    p.add_argument("--grade_accuracy", action='store_true', help="GPT 기반 정확도 평가 수행")
    p.add_argument("--grading_prompt_yaml", default="config/grading_prompt.yaml", help="정답 평가용 프롬프트 YAML 경로")
    p.add_argument("--evaluate_model", default="gpt-4o-mini", help="평가 모델 선택 (틱 1: gpt-4o-mini or gpt-4o)")
    p.add_argument("--num_workers", type=int, default=4, help="병렬 워커 수")
    
    return p.parse_args()


def load_connectors(yaml_path: str) -> Tuple[List[str], List[str]]:
    """
    YAML에서 CORRECT/INCORRECT 커넥터 리스트를 로드합니다.
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['CORRECT_CONNECTOR'], cfg['INCORRECT_CONNECTOR']


def load_data(path: str, mode: str):
    """
    데이터셋을 HF 또는 로컬에서 로드합니다.
    """
    if mode == 'hf':
        ds = load_dataset(path)['train']
    else:
        ds = load_from_disk(path)['train']
    return ds


def analyze_basic(ds, correct_conn: List[str], incorrect_conn: List[str]):
    """
    기본 메트릭(갯수, 평균 길이, 커넥터 및 final answer 출현 횟수)을 계산해 출력/로깅합니다.
    """
    print(ds)
    thinkings = ds['thinking_trajectories']
    answers = ds['attempt']

    failed = sum(1 for t in thinkings if 'Reasoning failed' in t)
    cnt = len(thinkings)
    avg_len = sum(len(t) for t in thinkings) / cnt
    cc = sum(any(c in t for c in correct_conn) for t in thinkings)
    ic = sum(any(icn in t for icn in incorrect_conn) for t in thinkings)
    fa = sum(1 for a in answers if 'final answer' in a.lower())

    # 터미널 출력
    print(f"Count: {cnt}, Failed: {failed}, AvgLen: {avg_len:.1f}")
    print(f"CorrectConn: {cc}, IncorrectConn: {ic}, FinalAnswer: {fa}")

     # 로깅
    logging.info(f"Total(전체 데이터의 개수): {cnt}")
    logging.info(f"Failed(Reasoning Failed 데이터의 개수): {failed}")
    logging.info(f"Avg reasoning length (평균 길이) {avg_len:.1f}")
    logging.info(f"CorrectConn hits (CORRECT CONNECTOR 개수): {cc}")
    logging.info(f"IncorrectConn hits (INCORRECT CONNECTOR 개수): {ic}")
    logging.info(f"FinalAnswers (FINAL ANSWER가 포함된 데이터의 개수): {fa}")


def grade_accuracy(ds, prompt_yaml: str, model: str, num_workers: int):
    """
    OpenAI GPT를 사용해 시도(attempt) vs 정답(solution) 비교 후 정확도를 계산합니다.
    """
    # YAML 로드
    with open(prompt_yaml, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    system_prompt = prompts['system_prompt']
    user_prompt = prompts['user_prompt']

    def grader(pair: Tuple[str, str]) -> bool:
        att, sol = pair

        from openai import OpenAI
        client = OpenAI()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(
                    attempt=att, solution=sol
                )}
            ]
        )
        return resp.choices[0].message.content.strip()

    batch = list(zip(ds['attempt'], ds['solution']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exe:
        results = list(tqdm(exe.map(grader, batch), total=len(batch), desc="Grading"))

    correct = sum(1 for r in results if r.strip().upper() == 'RIGHT')
    total = len(results)
    acc = correct / total * 100
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    logging.info(f"Grading accuracy: {acc:.2f}%")


def main():
    args = parse_args()
    log_file = setup_logging()

    correct_conn, incorrect_conn = load_connectors(args.connector_yaml)
    ds = load_data(args.dataset_dir, args.type)

    analyze_basic(ds, correct_conn, incorrect_conn)

    if args.grade_accuracy:
        grade_accuracy(
            ds,
            args.grading_prompt_yaml,
            args.evaluate_model,
            args.num_workers
        )

    print(f"Logs saved to: {log_file}")

if __name__ == '__main__':
    main()
