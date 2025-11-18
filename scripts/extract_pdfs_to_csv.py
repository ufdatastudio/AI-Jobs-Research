#!/usr/bin/env python3
"""
Extract all text from PDF files and populate a single CSV file.
Each PDF becomes one row in the CSV with full extracted text.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

import pdfplumber
import pandas as pd


def extract_all_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file without any parsing or cleaning."""
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"  ERROR extracting from {pdf_path.name}: {e}")
        return ""


def extract_title_from_filename(pdf_name: str) -> str:
    """Extract a reasonable job title from the PDF filename."""
    # Remove .pdf extension
    title = pdf_name.replace('.pdf', '')
    
    # Remove common suffixes
    title = re.sub(r'\s*-\s*Jobs.*$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*Careers.*$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*\([^)]*\)$', '', title)  # Remove parentheses
    title = re.sub(r'\s*Job Application for\s*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*in\s+[^,]+,\s+[^,]+$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'_\s*', ' ', title)  # Replace underscores with spaces
    title = re.sub(r'\s+', ' ', title)  # Normalize spaces
    
    return title.strip()


def extract_all_pdfs(data_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Extract all text from all PDFs in the data directory.
    Returns list of (pdf_name, job_title, full_text) tuples.
    """
    pdf_files = sorted(data_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"Processing [{i}/{len(pdf_files)}]: {pdf_path.name}")
        full_text = extract_all_text_from_pdf(pdf_path)
        if full_text:
            job_title = extract_title_from_filename(pdf_path.name)
            results.append((pdf_path.name, job_title, full_text))
            print(f"  Extracted {len(full_text)} characters")
        else:
            print(f"  WARNING: No text extracted from {pdf_path.name}")
    
    return results


def create_csv_from_pdfs(
    csv_path: Path,
    pdf_extractions: List[Tuple[str, str, str]]
):
    """
    Create a CSV file with extracted job postings from PDFs.
    Each PDF becomes one row with job_id, job_title, job_description, job_qualification, and job_posting.
    """
    # Prepare rows
    rows = []
    
    for i, (pdf_name, job_title, full_text) in enumerate(pdf_extractions, 1):
        # For now, put full text in both job_description and job_posting
        # job_qualification will be empty (can be extracted later if needed)
        row = {
            'job_id': str(i),
            'job_title': job_title,
            'job_description': full_text,
            'job_qualification': '',  # Empty for now, can be parsed later
            'job_posting': full_text  # Full text combined
        }
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\n✅ Created CSV: {csv_path}")
    print(f"   Total rows: {len(rows)}")
    print(f"   Total PDFs processed: {len(pdf_extractions)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all text from PDFs and create a single CSV file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data",
        help="Directory containing PDF files (default: Data)"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="Data/extracted_job_postings.csv",
        help="Path to output CSV file (default: Data/extracted_job_postings.csv)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv_path)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1
    
    # Extract from PDFs
    print("Extracting all text from PDFs...")
    pdf_extractions = extract_all_pdfs(data_dir)
    
    if not pdf_extractions:
        print("ERROR: No PDFs found or extraction failed")
        return 1
    
    # Create CSV
    print(f"\nCreating CSV file: {csv_path}")
    create_csv_from_pdfs(csv_path, pdf_extractions)
    
    return 0


if __name__ == "__main__":
    exit(main())