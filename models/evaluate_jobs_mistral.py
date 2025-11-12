from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

EVAL_PROMPT = """
You are an expert evaluator of cross-sector job postings related to Artificial Intelligence (AI) and Pedagogy in Engineering Education.

Evaluate the JOB POSTING below to determine how strongly it aligns with AI-related pedagogical work — i.e., roles that combine AI/ML with teaching, learning design, training, documentation, or educational enablement, as found across academia, edtech, industry, and government.

JOB POSTING:
{job_posting}

Rate each criterion from 1 (very poor / not present) to 5 (strongly evident):

1. Pedagogical Indicators – Mentions of teaching, designing, facilitating, assessing, documenting, or iterating learning activities, materials, or experiences (e.g., curriculum, tutorials, training, workshops, rubrics, or educational guides).
2. Learner / Audience Focus – Clear target learners such as students, educators, professional engineers, developers, or broader learning communities.
3. AI Relevance – Mentions of AI/ML, LLMs, GenAI, deep learning, or integration of AI tools in learning, training, or documentation contexts.
4. Responsible AI or Governance – References to ethics, fairness, transparency, safety, explainability, or responsible AI training and documentation.
5. Cross-Sector Context – Fit with academia, edtech, industry, or government roles that blend AI with learning, enablement, or curriculum development.
6. Overall Fit – How well this job represents the type of AI-pedagogical role described in the abstract (AI-related teaching, documentation, or educational enablement work).

Then apply this classification rule:

- Compute the mean of all six scores.
- If the mean score ≥ 3.5, set "ai_pedagogy_related": true.
- Otherwise, set "ai_pedagogy_related": false.
- Confidence should be proportional to the mean score (scale 0.0–1.0).

Respond STRICTLY in valid JSON (no markdown, comments, or extra text) as shown below:

Example:
{{
  "score_pedagogy": 5,
  "score_audience": 4,
  "score_ai_relevance": 5,
  "score_responsible_ai": 3,
  "score_cross_sector_context": 5,
  "score_overall_fit": 5,
  "ai_pedagogy_related": true,
  "confidence": 0.92,
  "rationale": "Strong evidence of AI-related learning design, developer enablement, and responsible-AI documentation across an edtech context."
}}
"""


# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load the Mistral judge model and tokenizer."""
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
def generate_evaluation(job_posting: str,
                        tokenizer,
                        model,
                        max_new_tokens: int = 512,
                        temperature: float = 0.2) -> str:
    """Generate JSON evaluation using Mistral as judge."""
    prompt = EVAL_PROMPT.format(job_posting=job_posting)

    messages = [
        {"role": "system", "content": "You are an expert evaluator of AI-pedagogy job postings. Always return valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    # Use chat template if available
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"System: You are an expert evaluator of AI-pedagogy job postings.\n\nUser: {prompt}\n\nAssistant:"

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
def load_csv(csv_path: str) -> List[Dict[str, str]]:
    """Load job postings from CSV file."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------
def evaluate_job_postings(csv_path: str,
                          job_posting_column: str,
                          output_dir: str = "results/JobPostings/Mistral",
                          tokenizer=None,
                          model=None,
                          max_new_tokens: int = 512,
                          temperature: float = 0.2):
    """Evaluate job postings to determine AI-pedagogy alignment using Mistral."""
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from CSV")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if job_posting_column not in rows[0]:
        print(f"WARNING: Column '{job_posting_column}' not found in CSV — skipping.")
        return

    print(f"\nEvaluating job postings from column: {job_posting_column}")
    evaluations = []

    for i, row in enumerate(rows):
        job_id = row.get("job_id", row.get("id", f"row_{i}"))
        job_posting = (row.get(job_posting_column) or "").strip()

        if not job_posting:
            continue

        print(f"  Evaluating {job_id} ({i+1}/{len(rows)})")
        try:
            response = generate_evaluation(job_posting, tokenizer, model,
                                           max_new_tokens=max_new_tokens, temperature=temperature)
            scores = extract_json_from_response(response)

            # Retry once if JSON fails
            if not scores:
                print(f"    WARNING: Retry due to malformed JSON ...")
                response = generate_evaluation(job_posting, tokenizer, model,
                                               max_new_tokens=max_new_tokens, temperature=temperature)
                scores = extract_json_from_response(response)

            if scores:
                evaluation = {
                    "job_id": job_id,
                    "job_posting": job_posting,
                    "judge_model": "mistral",
                    **scores
                }
                evaluations.append(evaluation)
            else:
                print(f"    ERROR: Failed to parse JSON for {job_id}")
                print(f"    Raw output snippet: {response[:200]}")

        except Exception as e:
            print(f"    ERROR: Error evaluating {job_id}: {e}")
            continue

        # Periodically clear cache to prevent OOM
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Save results
    json_out = output_path / "job_posting_evaluations.json"
    csv_out = output_path / "job_posting_evaluations.csv"
    pd.DataFrame(evaluations).to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(evaluations)} evaluations to:\n  {json_out}\n  {csv_out}")

    # Compute average scores
    if evaluations:
        score_keys = [
            "score_pedagogy", "score_audience", "score_ai_relevance",
            "score_responsible_ai", "score_cross_sector_context", "score_overall_fit"
        ]
        avg_scores = {}
        for key in score_keys:
            vals = [float(ev.get(key, 0)) for ev in evaluations if ev.get(key) is not None]
            avg_scores[key] = sum(vals) / len(vals) if vals else 0

        # Compute average confidence and ai_pedagogy_related percentage
        confidences = [float(ev.get("confidence", 0)) for ev in evaluations if ev.get("confidence") is not None]
        avg_scores["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0
        
        ai_related_count = sum(1 for ev in evaluations if ev.get("ai_pedagogy_related") == True)
        avg_scores["ai_pedagogy_related_percentage"] = (ai_related_count / len(evaluations) * 100) if evaluations else 0

        print(f"\nAverage scores:")
        for k, v in avg_scores.items():
            print(f"   {k:30s}: {v:.2f}")


# ---------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate job postings for AI-pedagogy alignment using Mistral as a judge.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to job postings CSV file.")
    parser.add_argument("--job_posting_column", type=str, default="job_posting", help="Column name containing job postings.")
    parser.add_argument("--output_dir", type=str, default="results/JobPostings/Mistral", help="Directory to save outputs.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Judge model ID.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_id)
    evaluate_job_postings(args.csv_path, args.job_posting_column, args.output_dir,
                          tokenizer, model, args.max_new_tokens, args.temperature)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

