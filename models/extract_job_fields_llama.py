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


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

EXTRACTION_PROMPT = """
You are an expert at extracting structured information from job postings.

Extract the following fields from the JOB POSTING text below:

1. **Job Title**: The official job title or position name (e.g., "Machine Learning Engineer", "AI Education Specialist")

2. **Job Description**: A clean, comprehensive overview of the role and its purpose. This should be a general description that introduces the position and its context within the organization. Remove any metadata, headers, footers, or navigation elements.

3. **Responsibilities**: Specific duties, tasks, and day-to-day activities the candidate will perform. This should be a list or description of what the person will actually do in this role (e.g., "Develop machine learning models", "Lead training workshops", "Collaborate with cross-functional teams"). If there's a "Responsibilities" or "What You'll Do" section, extract that content here.

4. **Required Skills**: Technical skills, tools, technologies, programming languages, frameworks, and soft skills that are necessary for this role. This can include both hard skills (e.g., "Python", "PyTorch", "AWS") and soft skills (e.g., "Excellent communication", "Team leadership"). Extract from sections like "Required Skills", "Skills", "Technical Requirements", or similar. If skills are mixed with qualifications, extract only the skill-related content here.

5. **Job Qualifications**: Education requirements, years of experience, certifications, degrees, and other formal requirements. This typically includes items like "Bachelor's degree", "5+ years of experience", "PhD preferred", etc. Include both "Required" and "Preferred" sections if present.

6. **Job Salary**: Salary range, compensation, or pay information if mentioned. If not explicitly stated, you may extract related compensation information (e.g., "competitive salary", "benefits package"). If no salary information is available, return an empty string.

JOB POSTING:
{job_posting}

Extract these fields accurately and cleanly. Remove any:
- Headers, footers, page numbers, dates
- Navigation elements or website metadata
- URLs or links
- Redundant or boilerplate text

Note: If a field is not explicitly present in the job posting (e.g., no separate "Responsibilities" section), you may leave it as an empty string. Focus on extracting what is clearly stated in the posting.

Respond STRICTLY in valid JSON (no markdown, comments, or extra text) as shown below:

Example:
{{
  "job_title": "Machine Learning Engineer",
  "job_description": "We are seeking an experienced Machine Learning Engineer to join our AI team. This role involves developing cutting-edge ML solutions...",
  "responsibilities": "Develop and deploy machine learning models for production use. Collaborate with data scientists and engineers. Design experiments and analyze results...",
  "required_skills": "Python, PyTorch, TensorFlow, AWS, Docker, Kubernetes, Strong problem-solving skills, Excellent communication",
  "job_qualification": "Required: Bachelor's degree in Computer Science or related field. 5+ years of ML experience. Preferred: Master's degree or PhD.",
  "job_salary": "$120,000 - $150,000 per year"
}}
"""


# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load the Llama model and tokenizer."""
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
# Core Extraction
# ---------------------------------------------------------------------
def generate_extraction(job_posting: str,
                       tokenizer,
                       model,
                       max_new_tokens: int = 1500,
                       temperature: float = 0.1) -> str:
    """Generate JSON extraction using Llama."""
    prompt = EXTRACTION_PROMPT.format(job_posting=job_posting)

    messages = [
        {"role": "system", "content": "You are an expert at extracting structured information from job postings. Always return valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    # Use chat template if available
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"System: You are an expert at extracting structured information from job postings.\n\nUser: {prompt}\n\nAssistant:"

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
# Main Extraction Function
# ---------------------------------------------------------------------
def extract_job_fields(csv_path: str,
                      job_posting_column: str,
                      output_path: str = "Data/extracted_job_fields_llama.csv",
                      tokenizer=None,
                      model=None,
                      max_new_tokens: int = 1500,
                      temperature: float = 0.1):
    """Extract structured job fields using Llama."""
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from CSV")

    if job_posting_column not in rows[0] if rows else {}:
        print(f"WARNING: Column '{job_posting_column}' not found in CSV — skipping.")
        return

    print(f"\nExtracting job fields from column: {job_posting_column}")
    extracted_jobs = []

    for i, row in enumerate(rows):
        job_id = row.get("job_id", row.get("id", f"row_{i+1}"))
        job_posting = (row.get(job_posting_column) or "").strip()

        if not job_posting:
            print(f"  Skipping {job_id} ({i+1}/{len(rows)}) — empty job posting")
            continue

        print(f"  Extracting fields from {job_id} ({i+1}/{len(rows)})")
        try:
            response = generate_extraction(job_posting, tokenizer, model,
                                          max_new_tokens=max_new_tokens, temperature=temperature)
            extracted_fields = extract_json_from_response(response)

            # Retry once if JSON fails
            if not extracted_fields:
                print(f"    WARNING: Retry due to malformed JSON ...")
                response = generate_extraction(job_posting, tokenizer, model,
                                              max_new_tokens=max_new_tokens, temperature=temperature)
                extracted_fields = extract_json_from_response(response)

            if extracted_fields:
                job_data = {
                    "job_id": job_id,
                    "job_title": extracted_fields.get("job_title", ""),
                    "job_description": extracted_fields.get("job_description", ""),
                    "responsibilities": extracted_fields.get("responsibilities", ""),
                    "required_skills": extracted_fields.get("required_skills", ""),
                    "job_qualification": extracted_fields.get("job_qualification", ""),
                    "job_salary": extracted_fields.get("job_salary", ""),
                }
                extracted_jobs.append(job_data)
            else:
                print(f"    ERROR: Failed to parse JSON for {job_id}")
                print(f"    Raw output snippet: {response[:200]}")
                # Add row with empty fields to maintain job_id sequence
                extracted_jobs.append({
                    "job_id": job_id,
                    "job_title": "",
                    "job_description": "",
                    "responsibilities": "",
                    "required_skills": "",
                    "job_qualification": "",
                    "job_salary": "",
                })

        except Exception as e:
            print(f"    ERROR: Error extracting fields from {job_id}: {e}")
            # Add row with empty fields to maintain job_id sequence
            extracted_jobs.append({
                "job_id": job_id,
                "job_title": "",
                "job_description": "",
                "responsibilities": "",
                "required_skills": "",
                "job_qualification": "",
                "job_salary": "",
            })
            continue

        # Periodically clear cache to prevent OOM
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(extracted_jobs)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

    print(f"\nSaved {len(extracted_jobs)} extracted job fields to: {output_file}")

    # Print summary statistics
    if extracted_jobs:
        non_empty_title = sum(1 for job in extracted_jobs if job.get("job_title", "").strip())
        non_empty_desc = sum(1 for job in extracted_jobs if job.get("job_description", "").strip())
        non_empty_resp = sum(1 for job in extracted_jobs if job.get("responsibilities", "").strip())
        non_empty_skills = sum(1 for job in extracted_jobs if job.get("required_skills", "").strip())
        non_empty_qual = sum(1 for job in extracted_jobs if job.get("job_qualification", "").strip())
        non_empty_salary = sum(1 for job in extracted_jobs if job.get("job_salary", "").strip())

        print(f"\nSummary Statistics:")
        print(f"   Total jobs processed: {len(extracted_jobs)}")
        print(f"   Jobs with title: {non_empty_title} ({non_empty_title/len(extracted_jobs)*100:.1f}%)")
        print(f"   Jobs with description: {non_empty_desc} ({non_empty_desc/len(extracted_jobs)*100:.1f}%)")
        print(f"   Jobs with responsibilities: {non_empty_resp} ({non_empty_resp/len(extracted_jobs)*100:.1f}%)")
        print(f"   Jobs with required skills: {non_empty_skills} ({non_empty_skills/len(extracted_jobs)*100:.1f}%)")
        print(f"   Jobs with qualifications: {non_empty_qual} ({non_empty_qual/len(extracted_jobs)*100:.1f}%)")
        print(f"   Jobs with salary info: {non_empty_salary} ({non_empty_salary/len(extracted_jobs)*100:.1f}%)")


# ---------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract structured job fields from job postings using Llama.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to job postings CSV file.")
    parser.add_argument("--job_posting_column", type=str, default="job_posting", help="Column name containing job postings.")
    parser.add_argument("--output_path", type=str, default="Data/extracted_job_fields_llama.csv", help="Output CSV file path.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Model ID.")
    parser.add_argument("--max_new_tokens", type=int, default=1500, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_id)
    extract_job_fields(args.csv_path, args.job_posting_column, args.output_path,
                      tokenizer, model, args.max_new_tokens, args.temperature)
    print("\nExtraction complete.")


if __name__ == "__main__":
    main()

