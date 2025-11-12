from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

EVAL_PROMPT = """
You are an expert Speech-Language Pathologist and evaluation judge for child-speech summarization.

Evaluate the MODEL SUMMARY below against the REFERENCE SUMMARY (created by Llama).

REFERENCE SUMMARY:
{reference}

MODEL SUMMARY:
{summary}

Rate each criterion from 1 (very poor) to 5 (excellent):

1. Overall summary quality
2. Fluency / readability / coherence
3. Faithfulness / factual correctness
4. Completeness / coverage of child's main points
5. Speaker purity (only the child's speech, no adult speech)
6. Usefulness for potential users (e.g., SLPs, parents, researchers)

Respond STRICTLY in valid JSON as shown below (no explanations or markdown):

Example:
{{
  "score_overall": 4,
  "score_fluency": 5,
  "score_faithfulness": 4,
  "score_coverage": 5,
  "score_purity": 5,
  "score_usefulness": 5,
  "rationale": "The summary is fluent and faithful but misses minor details."
}}
"""


# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load the Llama judge model and tokenizer."""
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------------------
def generate_evaluation(reference_summary: str,
                        model_summary: str,
                        tokenizer,
                        model,
                        max_new_tokens: int = 512,
                        temperature: float = 0.2) -> str:
    """Generate JSON evaluation using Llama as judge."""
    prompt = EVAL_PROMPT.format(reference=reference_summary, summary=model_summary)

    messages = [
        {"role": "system", "content": "You are an expert evaluation judge. Always return valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    # Use chat template if available
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"System: You are an expert evaluation judge.\n\nUser: {prompt}\n\nAssistant:"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------
# JSON Extraction Utility
# ---------------------------------------------------------------------
def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract the JSON object from model output text."""
    # Look for JSON code blocks
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------
def load_interview_csv(csv_path: str) -> List[Dict[str, str]]:
    """Load interview CSV data."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------
def evaluate_summaries(csv_path: str,
                       model_columns: List[str],
                       reference_column: str = "Llama_summary",
                       output_dir: str = "results/Interview/Evaluations",
                       tokenizer=None,
                       model=None,
                       max_new_tokens: int = 512,
                       temperature: float = 0.2):
    """Evaluate each model summary column against the reference summary."""
    rows = load_interview_csv(csv_path)
    print(f"✅ Loaded {len(rows)} rows from CSV")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_col in model_columns:
        if model_col not in rows[0]:
            print(f"⚠️ Column '{model_col}' not found in CSV — skipping.")
            continue

        print(f"\n🔍 Evaluating summaries from column: {model_col}")
        evaluations = []

        for i, row in enumerate(rows):
            audio_id = row.get("audio_id", f"row_{i}")
            reference = (row.get(reference_column) or "").strip()
            model_summary = (row.get(model_col) or "").strip()

            if not reference or not model_summary:
                continue

            print(f"  ▶ Evaluating {audio_id} ({i+1}/{len(rows)})")
            try:
                response = generate_evaluation(reference, model_summary, tokenizer, model,
                                               max_new_tokens=max_new_tokens, temperature=temperature)
                scores = extract_json_from_response(response)

                # Retry once if JSON fails
                if not scores:
                    print(f"    ⚠️ Retry due to malformed JSON ...")
                    response = generate_evaluation(reference, model_summary, tokenizer, model,
                                                   max_new_tokens=max_new_tokens, temperature=temperature)
                    scores = extract_json_from_response(response)

                if scores:
                    evaluation = {
                        "audio_id": audio_id,
                        "model_name": model_col,
                        "reference_summary": reference,
                        "model_summary": model_summary,
                        **scores
                    }
                    evaluations.append(evaluation)
                else:
                    print(f"    ❌ Failed to parse JSON for {audio_id}")
                    print(f"    Raw output snippet: {response[:200]}")

            except Exception as e:
                print(f"    ❌ Error evaluating {audio_id}: {e}")
                continue

            # Periodically clear cache to prevent OOM
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # Save results
        json_out = output_path / f"{model_col}_evaluations.json"
        csv_out = output_path / f"{model_col}_evaluations.csv"
        pd.DataFrame(evaluations).to_csv(csv_out, index=False)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved {len(evaluations)} evaluations to:\n  {json_out}\n  {csv_out}")

        # Compute average scores
        if evaluations:
            score_keys = [
                "score_overall", "score_fluency", "score_faithfulness",
                "score_coverage", "score_purity", "score_usefulness"
            ]
            avg_scores = {}
            for key in score_keys:
                vals = [float(ev.get(key, 0)) for ev in evaluations if ev.get(key) is not None]
                avg_scores[key] = sum(vals) / len(vals) if vals else 0

            print(f"\n📊 Average scores for {model_col}:")
            for k, v in avg_scores.items():
                print(f"   {k:20s}: {v:.2f}")


# ---------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate model summaries using Llama as a judge.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to interview CSV file.")
    parser.add_argument("--model_columns", nargs="+", default=["AF3_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary"],
                        help="Model summary columns to evaluate.")
    parser.add_argument("--reference_column", type=str, default="Llama_summary", help="Reference summary column.")
    parser.add_argument("--output_dir", type=str, default="results/Interview/Evaluations", help="Directory to save outputs.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Judge model ID.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_id)
    evaluate_summaries(args.csv_path, args.model_columns, args.reference_column, args.output_dir,
                       tokenizer, model, args.max_new_tokens, args.temperature)
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
