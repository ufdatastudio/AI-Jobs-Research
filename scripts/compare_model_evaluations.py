#!/usr/bin/env python3
"""
Compare evaluation results from three LLM models (Llama, Qwen2, Mistral).
Calculates correlation coefficients and creates a summary CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import csv


def load_evaluation_results(csv_path: Path) -> pd.DataFrame:
    """Load evaluation results CSV and extract key columns."""
    df = pd.read_csv(csv_path)
    # Select only the columns we need: job_id, mean_score, ai_pedagogy_related
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
        # Handle string representations
        if value.lower() in ['true', '1', 'yes']:
            return 1
        elif value.lower() in ['false', '0', 'no']:
            return 0
    # Try to convert to bool then int
    try:
        return 1 if bool(value) else 0
    except:
        return 0


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Calculate Pearson correlation coefficients between model mean scores."""
    correlations = {}
    
    models = ['llama', 'qwen2', 'mistral']
    mean_cols = [f'{model}_mean' for model in models]
    
    # Check which columns exist
    available_cols = [col for col in mean_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("Warning: Need at least 2 models to calculate correlations.")
        return correlations
    
    # Calculate pairwise correlations
    for i, model1 in enumerate(models):
        col1 = f'{model1}_mean'
        if col1 not in df.columns:
            continue
            
        for model2 in models[i+1:]:
            col2 = f'{model2}_mean'
            if col2 not in df.columns:
                continue
            
            # Get valid pairs (non-null values)
            valid_df = df[[col1, col2]].dropna()
            
            if len(valid_df) < 2:
                print(f"Warning: Not enough data for {model1}-{model2} correlation.")
                continue
            
            corr_coef, p_value = pearsonr(valid_df[col1], valid_df[col2])
            key = f"{model1}_vs_{model2}"
            correlations[key] = {
                'correlation': corr_coef,
                'p_value': p_value,
                'n': len(valid_df)
            }
    
    return correlations


def main():
    base_dir = Path("/orange/ufdatastudios/c.okocha/AI-Jobs-Research")
    
    # Paths to evaluation results
    llama_path = base_dir / "results/JobPostings/Llama/job_posting_evaluations.csv"
    qwen2_path = base_dir / "results/JobPostings/Qwen2/job_posting_evaluations.csv"
    mistral_path = base_dir / "results/JobPostings/Mistral/job_posting_evaluations.csv"
    
    # Load evaluation results
    print("Loading evaluation results...")
    llama_df = load_evaluation_results(llama_path)
    llama_df = llama_df.rename(columns={
        'mean_score': 'llama_mean',
        'ai_pedagogy_related': 'llama_ai_pedagogy'
    })
    
    qwen2_df = load_evaluation_results(qwen2_path)
    qwen2_df = qwen2_df.rename(columns={
        'mean_score': 'qwen2_mean',
        'ai_pedagogy_related': 'qwen2_ai_pedagogy'
    })
    
    mistral_df = load_evaluation_results(mistral_path)
    mistral_df = mistral_df.rename(columns={
        'mean_score': 'mistral_mean',
        'ai_pedagogy_related': 'mistral_ai_pedagogy'
    })
    
    # Merge all three dataframes on job_id
    print("Merging evaluation results...")
    merged_df = llama_df.merge(qwen2_df, on='job_id', how='outer')
    merged_df = merged_df.merge(mistral_df, on='job_id', how='outer')
    
    # Sort by job_id for consistency
    merged_df = merged_df.sort_values('job_id').reset_index(drop=True)
    
    # Convert boolean columns to int (0 or 1) for easier analysis
    for col in ['llama_ai_pedagogy', 'qwen2_ai_pedagogy', 'mistral_ai_pedagogy']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].apply(convert_boolean_to_int)
    
    # Calculate correlations
    print("\nCalculating correlation coefficients...")
    correlations = calculate_correlations(merged_df)
    
    # Print correlation results
    print("\n" + "="*70)
    print("CORRELATION COEFFICIENTS BETWEEN MODELS")
    print("="*70)
    for pair, stats in correlations.items():
        print(f"{pair:30s}: r = {stats['correlation']:.4f}, p = {stats['p_value']:.4f}, n = {stats['n']}")
    print("="*70)
    
    # Create summary CSV with just the essential columns
    summary_columns = ['job_id']
    for model in ['llama', 'qwen2', 'mistral']:
        if f'{model}_mean' in merged_df.columns:
            summary_columns.append(f'{model}_mean')
        if f'{model}_ai_pedagogy' in merged_df.columns:
            summary_columns.append(f'{model}_ai_pedagogy')
    
    summary_df = merged_df[summary_columns].copy()
    
    # Save summary CSV
    output_path = base_dir / "Data/model_comparison_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\nSummary CSV saved to: {output_path}")
    print(f"Total jobs in comparison: {len(summary_df)}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for col in summary_df.columns:
        if col == 'job_id':
            continue
        if 'mean' in col:
            mean_val = summary_df[col].mean()
            std_val = summary_df[col].std()
            print(f"{col:30s}: Mean = {mean_val:.3f}, Std = {std_val:.3f}")
        elif 'ai_pedagogy' in col:
            true_count = summary_df[col].sum()
            total_count = summary_df[col].notna().sum()
            percentage = (true_count / total_count * 100) if total_count > 0 else 0
            print(f"{col:30s}: {true_count}/{total_count} ({percentage:.1f}%) classified as AI-pedagogy related")
    print("="*70)
    
    # Save correlation results to a text file
    corr_output_path = base_dir / "Data/model_correlations.txt"
    with open(corr_output_path, 'w') as f:
        f.write("CORRELATION COEFFICIENTS BETWEEN MODELS\n")
        f.write("="*70 + "\n")
        for pair, stats in correlations.items():
            f.write(f"{pair:30s}: r = {stats['correlation']:.4f}, p = {stats['p_value']:.4f}, n = {stats['n']}\n")
        f.write("="*70 + "\n")
    
    print(f"\nCorrelation results saved to: {corr_output_path}")


if __name__ == "__main__":
    main()

