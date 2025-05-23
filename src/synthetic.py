import os
import time
import argparse
import yaml 
import logging
import numpy as np
import concurrent.futures

from google import genai
from datasets import load_dataset, Dataset, DatasetDict
from typing import Tuple, List
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def parse_args() -> argparse.Namespace:
    """
    CLI 인자를 파싱합니다.

    --dataset_name: Hugging Face 데이터셋 이름 또는 경로
    --split: 사용할 데이터셋 스플릿
    --prompt_yaml: 프롬프트 템플릿 YAML 파일 경로
    --connector_yaml: 커넥터 리스트 YAML 파일 경로
    --output_dir: 결과 출력 디렉토리
    --sample_size: 샘플링할 데이터 개수 (선택)
    --num_workers: 병렬 작업자 수
    """
    p = argparse.ArgumentParser(description="Generate CAC-CoT synthetic dataset.")

    # -- DATA
    p.add_argument("--dataset_name", default="datumo/datumo-gemini-short-v2", help="HuggingFace 데이터 파일을 로드합니다.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")

    # -- PROMPT
    p.add_argument("--prompt_yaml", default="config/synthetic_prompt.yaml", help="프롬프트 템플릿 YAML 파일 경로")
    p.add_argument("--version", default='v4', help="config/synthetic_prompt.yaml 파일에서 사용할 프롬프트를 선택")
    p.add_argument("--systemprompt_yaml", default="config/system_prompt.yaml", help="학습용 시스템 프롬프트 템플릿 YAML 파일 경로")
    p.add_argument("--system_key", default='qwen', help="config/system_prompt.yaml 파일에서 사용할 프롬프트를 선택")

    # -- CONNECTOR
    p.add_argument("--connector_yaml", default="config/connector.yaml", help="커넥터 리스트 YAML 파일 경로")

    # -- SAVE
    p.add_argument("--output_dir", required=True, help="출력 디렉토리 경로")
    p.add_argument("--push_to_hub", action="store_true", help="허깅페이스에 데이터 푸시, (output_dir)로 저장")
    p.add_argument("--token", default="", help="push_to_hub=True(defulat: privacy) 일 경우, 개인 huggingface token 입력")

    # -- CONSTRAINTS
    p.add_argument("--constraint_min_length", default=100, type=int, help="생성된 데이터 길이제약 최소값")
    p.add_argument("--constraint_max_length", default=30000, type=int, help="생성된 데이터 길이제약 최소값")

    # --  TEST
    p.add_argument("--test", action="store_true", help="샘플링 여부")
    p.add_argument("--sample_size", type=int, help="샘플링할 데이터 수 (선택)")

    # -- EXECUTION 
    p.add_argument("--num_workers", type=int, default=4, help="병렬 쓰레드 수")
    
    return p.parse_args()


def build_prompt(question: str,
                 prompt_tmpl: str,
                 thinking_start: str,
                 thinking_end: str,
                 answer_start: str,
                 answer_end: str,
                 correct_connector: List[str],
                 incorrect_connector: List[str]) -> str:
    """
    Gemini API 요청용 프롬프트를 구성합니다.

    Inputs:
      - question: 문제 텍스트
      - prompt_tmpl: 템플릿 문자열
      - thinking_start/end: 추론 시작/종료 토큰
      - answer_start/end: 답변 시작/종료 토큰
      - correct/incorrect_connector: 맞았을/틀렸을 때 사용하는 연결어 리스트

    Returns:
      - API 호출에 사용할 최종 프롬프트
    """
    return prompt_tmpl.format(
        thinking_start_token=thinking_start,
        thinking_end_token=thinking_end,
        answer_start_token=answer_start,
        answer_end_token=answer_end,
        question=question,
        correct_connector=correct_connector,
        incorrect_connector=incorrect_connector
    )


def setup_logging(log_dir: str = "log", prefix: str = "constraint") -> str:
    """
    로그 파일을 지정된 폴더에 현재 시간 기반 이름으로 생성하고 로깅 설정을 수행합니다.

    Args:
        log_dir (str): 로그 파일이 저장될 디렉토리 이름
        prefix (str): 로그 파일 이름 앞에 붙을 접두사

    Returns:
        str: 생성된 로그 파일의 전체 경로
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    logging.basicConfig(
        filename=log_filename,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("로깅이 초기화되었습니다.")


def gemini_qa(prompt: str,
              model_name: str = "gemini-2.0-flash") -> str:
    """
    Gemini API에 쿼리하여 텍스트를 생성합니다.

    Inputs:
      - prompt: 완성된 프롬프트 문자열
      - model_name: 사용할 모델 이름

    Returns:
      - API 응답 텍스트
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    res = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return res.text if hasattr(res, "text") else str(res)


def constraint_1(response: str, min_length: int, max_length: int) -> str:
    """ 데이터 생성 시, 길이 제약 """
    if len(response) >= min_length and len(response) <= max_length:
        return response
    logging.warning(f"[Constraint 1 Violation] Length out of bounds: {len(response)} (Expected {min_length}~{max_length})\n")
    return None


def constraint_2(response: str, ASTARTT: str, AENDT: str, TSTARTT: str, TENDT: str) -> Tuple[str, str]:
    """ 데이터 생성 시, THINKING/ANSWER TOKEN FORMAT 제약 """
    if (ASTARTT in response and AENDT in response and TSTARTT in response and TENDT in response):
        thinking = response.split(TSTARTT)[1].split(TENDT)[0].strip()
        answer = response.split(ASTARTT)[1].split(AENDT)[0].strip()
        return thinking, answer
    logging.warning(f"[Constraint 2 Violation] Missing tokens in response\n")
    return None, None


def constraint_3(thinking: str, answer: str) -> bool:
    """ 데이터 생성 시, 잘못된 위치 파싱 제약 """
    if '<answer>' in thinking.lower():
        logging.warning(f"[Constraint 3 Violation] '<answer>' found in thinking\n")
        return False
    
    if 'think' in answer.lower():
        logging.warning(f"[Constraint 3 Violation] '<answer>' found in thinking\n")
        return False
    
    return True


def main():
    """
    실행 흐름:
    1. CLI 인자 파싱
    2. YAML 설정 로드 (프롬프트 템플릿, 커넥터 리스트)
    3. Gemini API 키 설정 확인
    4. 데이터셋 로드 및 샘플링
    5. 병렬로 생성 수행 (추론 + 답변)
    6. 실패 항목 필터링 후 저장
    """
    args = parse_args()
    setup_logging()

    # YAML 파일에서 프롬프트 템플릿 로드
    with open(args.prompt_yaml, 'r', encoding='utf-8') as f:
        prompt_cfg = yaml.safe_load(f)
    PROMPT_TMPL: str = prompt_cfg[args.version]

    with open(args.systemprompt_yaml, 'r', encoding='utf-8') as f:
        system_prompt = yaml.safe_load(f)
    SYSTEM_PROMPT: str = system_prompt[args.system_key]

    # YAML 파일에서 커넥터 리스트 로드
    with open(args.connector_yaml, 'r', encoding='utf-8') as f:
        conn_cfg = yaml.safe_load(f)
    CORRECT_CONNECTOR: List[str] = conn_cfg['CORRECT_CONNECTOR']
    INCORRECT_CONNECTOR: List[str] = conn_cfg['INCORRECT_CONNECTOR']

    # Gemini API 키 설정
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("ENV 변수 GEMINI_API_KEY를 설정해주세요.")

    # 데이터셋 로드 및 샘플링
    ds = load_dataset(args.dataset_name)
    ds = ds['train']
    if args.test:
        ds = ds.shuffle(seed=42).select(range(args.sample_size))

    q_list = ds["question"]
    sol_list = ds["solution"]

    results = [None] * len(q_list)
    drops: List[int] = []

    def worker(item: Tuple[int, str]) -> Tuple[int, str, str, bool]:
        """
        병렬 처리 워커 함수

        Inputs:
          - item: (인덱스, 질문)

        Returns:
          - 인덱스, 추론 텍스트, 답변 텍스트, 드롭 여부
        """
        idx, question = item
        prompt = build_prompt(
            question,
            PROMPT_TMPL,
            TSTARTT, TENDT,
            ASTARTT, AENDT,
            CORRECT_CONNECTOR,
            INCORRECT_CONNECTOR
        )
        for _ in range(5):
            resp = gemini_qa(prompt)
            if not constraint_1(resp, args.constraint_min_length, args.constraint_max_length):
                continue

            thinking, answer = constraint_2(resp, ASTARTT, AENDT, TSTARTT, TENDT)
            if thinking is None:
                continue

            if not constraint_3(thinking, answer):
                continue
            
            return idx, thinking, answer, False
        return idx, "", "", True

    # 6) 병렬 생성 수행
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for idx, thinking, answer, dropped in tqdm(
                executor.map(worker, enumerate(q_list)),
                total=len(q_list), desc="Generating"):
            if dropped:
                drops.append(idx)
            results[idx] = (thinking, answer)

    # 7) 드롭된 항목 제외 후 저장
    def tokenized(questions: List[str], thinkings: List[str], answers: List[str], system_prompt: str, model: str):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)

        texts = []
        for q, t, a in zip(questions, thinkings, answers):
            text = tokenizer.apply_chat_template([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': q},
                {'role': 'assistant', 'content': f"<|im_start|>think\n{t.strip()}" + f"\n<|im_start|>answer\n{a.strip()}"}
            ], tokenize=False)
        
            texts.append(text)
        
        return texts

    keep = [i for i in range(len(results)) if i not in drops]
    q_list = [q_list[i] for i in keep]
    sol_list = [sol_list[i] for i in keep]
    thinkings = [results[i][0] for i in keep]
    attempts = [results[i][1] for i in keep]
    out_ds = DatasetDict({
        "train": Dataset.from_dict({
            "question": q_list,
            "solution": sol_list,
            "thinking_trajectories": thinkings,
            "attempt": attempts,
            "texts": tokenized(q_list, thinkings, attempts, SYSTEM_PROMPT, args.model)
        })
    })

    if args.push_to_hub:
        out_ds.push_to_hub(
            args.output_dir,
            token=args.token,
            private=True
        )
        logging.info(f"Push to hub 성공: {args.output_dir}")
    else:
        out_ds.save_to_dist(args.output_dir)
        logging.info(f"Save to disk 완료: {args.output_dir}")

    print(f"Completed: Saved {len(keep)} examples to {args.output_dir}. Dropped {len(drops)}.")

    
if __name__ == "__main__":
    
    # 토큰 마커
    TSTARTT, TENDT = "<thinking>", "</thinking>"
    ASTARTT, AENDT = "<answer>", "</answer>"

    # logging 파일 설정
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join("log", f"constraint_{timestamp}.log")

    logging.basicConfig(
        filename=log_filename,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    main()
