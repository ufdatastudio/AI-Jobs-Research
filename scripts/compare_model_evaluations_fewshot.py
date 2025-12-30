#!/usr/bin/env python3
"""
Compare few-shot evaluation results from five LLM models (Gemma, Llama, Mistral, OLMo, Qwen).
Calculates correlation coefficients and creates a summary CSV, mirroring zero-shot comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import csv


def load_evaluation_results(csv_path: Path) -> pd.DataFrame:
    """Load evaluation results CSV and extract key columns."""
    df = pd.read_csv(csv_path)
    key_columns = ['job_id', 'mean_score', 'ai_pedagogy_related']
    if not all(col in df.columns for col in key_columns):
        raise ValueError(f"Missing required columns in {csv_path}. Found: {df.columns.tolist()}")
    return df[key_columns].copy()


def convert_boolean_to_int(value):
    """Convert boolean or string boolean to int (0 or 1)."""
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        if value.lower() in ['true', '1', 'yes']:
            return 1
        if value.lower() in ['false', '0', 'no']:
            return 0
    try:
        return 1 if bool(value) else 0
    except Exception:
        return 0


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Calculate Pearson correlation coefficients between model mean scores."""
    correlations = {}
    models = ['gemma', 'llama', 'mistral', 'olmo', 'qwen']
    mean_cols = [f'{model}_mean' for model in models]
    available_cols = [col for col in mean_cols if col in df.columns]
    if len(available_cols) < 2:
        print("Warning: Need at least 2 models to calculate correlations.")
        return correlations
    for i, model1 in enumerate(models):
        col1 = f'{model1}_mean'
        if col1 not in df.columns:
            continue
        for model2 in models[i + 1:]:
            col2 = f'{model2}_mean'
            if col2 not in df.columns:
                continue
            valid_df = df[[col1, col2]].dropna()
            if len(valid_df) < 2:
                print(f"Warning: Not enough data for {model1}-{model2} correlation.")
                continue
            corr_coef, p_value = pearsonr(valid_df[col1], valid_df[col2])
            correlations[f"{model1}_vs_{model2}"] = {
                'correlation': round(corr_coef, 6),
                'p_value': float(p_value),
                'n': int(len(valid_df)),
            }
    return correlations


def main():
    base_dir = Path("/orange/ufdatastudios/c.okocha/AI-Jobs-Research")

    model_paths = {
        'gemma': base_dir / "results/JobPostings/Gemma_FewShot" / "job_posting_evaluations_fewshot.csv",
        'llama': base_dir / "results/JobPostings/Llama_FewShot" / "job_posting_evaluations_fewshot.csv",
        'mistral': base_dir / "results/JobPostings/Mistral_FewShot" / "job_posting_evaluations_fewshot.csv",
        'olmo': base_dir / "results/JobPostings/OLMo_FewShot" / "job_posting_evaluations_fewshot.csv",
        'qwen': base_dir / "results/JobPostings/Qwen_FewShot" / "job_posting_evaluations_fewshot.csv",
    }

    print("Loading few-shot evaluation results...")
    model_dfs = {}
    for model_name, csv_path in model_paths.items():
        if csv_path.exists():
            try:
                df = load_evaluation_results(csv_path)
                df = df.rename(columns={
                    'mean_score': f'{model_name}_mean',
                    'ai_pedagogy_related': f'{model_name}_ai_pedagogy'
                })
                model_dfs[model_name] = df
                print(f"  Loaded {model_name}: {len(df)} jobs")
            except Exception as e:
                print(f"  WARNING: Failed to load {model_name}: {e}")
        else:
            print(f"  WARNING: {csv_path} not found")

    if len(model_dfs) == 0:
        print("ERROR: No few-shot evaluation results found!")
        return

    print("\nMerging few-shot evaluation results...")
    merged_df = None
    for model_name, df in model_dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on='job_id', how='outer')

    merged_df = merged_df.sort_values('job_id').reset_index(drop=True)

    ai_pedagogy_cols = [col for col in merged_df.columns if 'ai_pedagogy' in col]
    for col in ai_pedagogy_cols:
        merged_df[col] = merged_df[col].apply(convert_boolean_to_int)

    print("\nCalculating correlation coefficients (few-shot)...")
    correlations = calculate_correlations(merged_df)

    print("\n" + "=" * 70)
    print("FEW-SHOT CORRELATION COEFFICIENTS BETWEEN MODELS")
    print("=" * 70)
    for pair, stats in correlations.items():
        print(f"{pair:30s}: r = {stats['correlation']:.4f}, p = {stats['p_value']:.4f}, n = {stats['n']}")
    print("=" * 70)

    summary_columns = ['job_id']
    for model in ['gemma', 'llama', 'mistral', 'olmo', 'qwen']:
        if f'{model}_mean' in merged_df.columns:
            summary_columns.append(f'{model}_mean')
        if f'{model}_ai_pedagogy' in merged_df.columns:
            summary_columns.append(f'{model}_ai_pedagogy')

    summary_df = merged_df[summary_columns].copy()
    output_path = base_dir / "Data/model_comparison_fewshot_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nFew-shot summary CSV saved to: {output_path}")
    print(f"Total jobs in comparison (few-shot): {len(summary_df)}")

    corr_output_path = base_dir / "Data/model_correlations_fewshot.txt"
    with open(corr_output_path, 'w') as f:
        f.write("FEW-SHOT: CORRELATION COEFFICIENTS BETWEEN MODELS\n")
        f.write("=" * 70 + "\n")
        for pair, stats in correlations.items():
            f.write(f"{pair:30s}: r = {stats['correlation']:.4f}, p = {stats['p_value']:.4f}, n = {stats['n']}\n")
        f.write("=" * 70 + "\n")
        f.write("\nFEW-SHOT SUMMARY STATISTICS\n")
        f.write("=" * 70 + "\n")
        for col in summary_df.columns:
            if col == 'job_id':
                continue
            if 'mean' in col:
                mean_val = summary_df[col].mean()
                std_val = summary_df[col].std()
                f.write(f"{col:30s}: Mean = {mean_val:.3f}, Std = {std_val:.3f}\n")
            elif 'ai_pedagogy' in col:
                true_count = summary_df[col].sum()
                total_count = summary_df[col].notna().sum()
                percentage = (true_count / total_count * 100) if total_count > 0 else 0
                f.write(f"{col:30s}: {true_count}/{total_count} ({percentage:.1f}%) classified as AI-pedagogy related\n")
        f.write("=" * 70 + "\n")

    print(f"\nFew-shot correlation results saved to: {corr_output_path}")


if __name__ == "__main__":
    main()


