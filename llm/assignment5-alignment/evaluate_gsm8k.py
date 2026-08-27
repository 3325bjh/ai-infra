from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI

from cs336_alignment.parse_utils import parse_gsm8k_response

VLLM_API_URL = "http://localhost:8010/v1"
MODEL_NAME = "Qwen2.5-Math-1.5B-Base"
DATA_PATH = Path("data/gsm8k/test.jsonl")
OUTPUT_FILE = Path(f"result/gsm8k_{MODEL_NAME}_baseline_results.json")
MAX_WORKERS = 2
client = OpenAI(base_url=VLLM_API_URL, api_key="EMPTY")

SYSTEM_PROMPT = (
    "Below is a list of conversations between a human and an AI assistant (you).\n"
    'Users place their queries under "# Query:", and your responses are under "# Answer:".\n'
    "You are a helpful, respectful, and honest assistant.\n"
    "You should always answer as helpfully as possible while ensuring safety.\n"
    "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
    "Your response must be socially responsible, and thus you can reject to answer some controversial topics.\n"
)

def load_gsm8k_data(file_path: Path) -> list[dict]:
    items = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            user_query = f"{ex['question']}\nAnswer:"
            full_prompt = (
                f"{SYSTEM_PROMPT}\n\n#Query:\n```\n{user_query}\n```\n\n#Answer:\n```\n"
            )
            gold_str = ex["answer"].split("####")[-1].strip().replace(",", "")
            items.append({
                "prompt": full_prompt,
                "gold": float(gold_str),
                "original_question": ex["question"],
            })
    return items


def call_api(item: dict) -> dict:
    try:
        response=client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": item["prompt"]}],
            temperature=0.0,
            max_tokens=512,
        )
        gen_text = response.choices[0].message.content or ""
        pred = parse_gsm8k_response(gen_text)
        try:
            is_correct = pred is not None and float(pred) == item["gold"]
        except ValueError:
            is_correct = False
        return {
            "question": item["original_question"],
            "gold": item["gold"],
            "pred": pred,
            "output": gen_text,
            "is_correct": is_correct,
        }
    except Exception as e:
        return {"error": str(e), "question": item["original_question"]}

def main() -> None:
    all_items = load_gsm8k_data(DATA_PATH)
    all_results = []
    errors = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(call_api, item) for item in all_items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result = future.result()
            if "error" in result:
                errors.append(result)
            else:
                all_results.append(result)

    duration = time.time() - start_time
    correct = sum(result["is_correct"] for result in all_results)
    parse_failed = sum(result["pred"] is None for result in all_results)
    metrics = {
        "accuracy": correct / len(all_results) if all_results else 0.0,
        "throughput_ex_per_sec": len(all_results) / duration if duration else 0.0,
        "parsing_failure_count": parse_failed,
        "successful_examples": len(all_results),
        "failed_examples": len(errors),
        "total_examples": len(all_items),
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": all_results, "errors": errors}, f, indent=2)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Saved results to {OUTPUT_FILE}")



if __name__ == '__main__':
    main()
