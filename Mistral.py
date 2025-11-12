from __future__ import annotations
import argparse, csv, json, re, torch, pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------
#  MODEL CONFIGURATION
# ---------------------------------------------------------------------
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


# ---------------------------------------------------------------------
#  EVALUATION PROMPT
# ---------------------------------------------------------------------
EVAL_PROMPT = """
You are an expert Speech-Language Pathologist and evaluation judge for child reading summarization.

Evaluate the MODEL SUMMARY below against the REFERENCE SUMMARY (created by Llama).

REFERENCE SUMMARY:
{reference}

MODEL SUMMARY:
{summary}

Rate each criterion from 1 (very poor) to 5 (excellent):

1. Overall summary quality
2. Fluency / readability / coherence
3. Faithfulness / factual correctness
4. Completeness / coverage of the child's main points
5. Usefulness for potential users (e.g., teachers, researchers, SLPs)

Respond STRICTLY in valid JSON as shown below (no explanations or markdown):

Example:
{{
  "score_overall": 4,
  "score_fluency": 5,
  "score_faithfulness": 4,
  "score_coverage": 5,
  "score_usefulness": 5,
  "rationale": "The summary is fluent, accurate, and clear but misses a few supporting details."
}}
"""


# ---------------------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str = MODEL_ID):
    print(f"Loading model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    mdl.eval()
    return tok, mdl


def generate_evaluation(reference: str, summary: str, tokenizer, model,
                        max_new_tokens=512, temperature=0.2) -> str:
    prompt = EVAL_PROMPT.format(reference=reference, summary=summary)
    messages = [
        {"role": "system", "content": "You are an expert evaluation judge. Always return valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"System: You are an expert evaluation judge.\n\nUser: {prompt}\n\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def extract_json_from_response(text: str) -> Optional[Dict]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
#  EVALUATION CORE
# ---------------------------------------------------------------------
def evaluate_with_mistral(csv_path: str,
                        model_columns: List[str], reference_column: str,
                        output_dir: str, max_new_tokens=512, temperature=0.2):
    """Evaluate using Mistral as judge."""
    rows = load_csv(csv_path)
    print(f"\n🔹 Evaluating with Mistral ({len(rows)} rows)")
    tok, mdl = load_model_and_tokenizer(MODEL_ID)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for model_col in model_columns:
        if model_col not in rows[0]:
            print(f"Column {model_col} missing, skipping.")
            continue
        evals = []
        for i, row in enumerate(rows):
            ref = (row.get(reference_column) or "").strip()
            summ = (row.get(model_col) or "").strip()
            if not ref or not summ:
                continue
            try:
                resp = generate_evaluation(ref, summ, tok, mdl,
                                           max_new_tokens=max_new_tokens,
                                           temperature=temperature)
                scores = extract_json_from_response(resp)
                if not scores:  # retry once
                    resp = generate_evaluation(ref, summ, tok, mdl,
                                               max_new_tokens=max_new_tokens,
                                               temperature=temperature)
                    scores = extract_json_from_response(resp)
                if scores:
                    evals.append({
                        "judge_model": "mistral",
                        "audio_id": row.get("audio_id", f"row_{i}"),
                        "model_name": model_col,
                        **scores
                    })
            except Exception as e:
                print(f"   {model_col} row {i}: {e}")
                continue
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        out_csv = Path(output_dir) / f"{model_col}_mistral_eval.csv"
        out_json = Path(output_dir) / f"{model_col}_mistral_eval.json"
        pd.DataFrame(evals).to_csv(out_csv, index=False)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(evals, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(evals)} evaluations to {out_csv}")

        # Quick average reporting
        if evals:
            keys = ["score_overall","score_fluency","score_faithfulness",
                    "score_coverage","score_usefulness"]
            avg = {k: sum(float(e.get(k,0)) for e in evals)/len(evals) for k in keys}
            print(f"Avg ({model_col}, mistral): " +
                  ", ".join(f"{k}:{v:.2f}" for k,v in avg.items()))


# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate model summaries using Mistral as judge.")
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--model_columns", nargs="+",
                    default=["AF35_summary","Qwen_summary","Kimi_summary","Salmon_summary","GAMA_summary"])
    ap.add_argument("--reference_column", default="LlamaRead_summary")
    ap.add_argument("--output_dir", default="results/Reading/Mistral")
    ap.add_argument("--model_id", type=str, default=MODEL_ID, help="Mistral model ID")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    try:
        evaluate_with_mistral(args.csv_path, args.model_columns,
                           args.reference_column, args.output_dir,
                           args.max_new_tokens, args.temperature)
        torch.cuda.empty_cache()
        print(f"✅ Completed Mistral evaluation")
    except Exception as e:
        print(f"❌ Failed Mistral evaluation: {e}")
        raise


if __name__ == "__main__":
    main()

