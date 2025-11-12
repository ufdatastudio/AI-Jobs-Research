# AI Jobs Research: Classification of AI in Education Job Postings

A research project that employs Large Language Models (LLMs) as evaluators to classify job postings as AI-related pedagogical positions. This project aims to systematically identify and analyze roles that combine Artificial Intelligence/Machine Learning with teaching, learning design, training, documentation, or educational enablement across academia, edtech, industry, and government sectors.

## Overview

This project addresses the growing need to identify and classify job postings related to AI in Education—specifically roles that involve pedagogical work combining AI/ML technologies with educational practices. The evaluation framework uses LLM-based judges to assess job postings across multiple criteria, providing a scalable and consistent approach to classification.

### Research Goals

- **Automated Classification**: Use LLM judges to evaluate job postings for AI-pedagogy alignment
- **Multi-Annotator Approach**: Combine multiple LLM evaluators and human annotators for robust classification
- **Cross-Sector Analysis**: Identify AI-education roles across academia, edtech, industry, and government
- **Evaluation Framework**: Establish criteria and metrics for assessing job posting relevance

## Methodology

### Evaluation Framework

The project uses a structured evaluation framework where LLM judges assess job postings across six key criteria:

1. **Pedagogical Indicators** (1-5): Evidence of teaching, designing, facilitating, assessing, documenting, or iterating learning activities, materials, or experiences
2. **Learner/Audience Focus** (1-5): Clear identification of target learners (students, educators, professional engineers, developers, etc.)
3. **AI Relevance** (1-5): Mentions of AI/ML, LLMs, GenAI, deep learning, or integration of AI tools in learning contexts
4. **Responsible AI or Governance** (1-5): References to ethics, fairness, transparency, safety, explainability, or responsible AI practices
5. **Cross-Sector Context** (1-5): Alignment with academia, edtech, industry, or government roles that blend AI with learning
6. **Overall Fit** (1-5): How well the job represents AI-pedagogical roles (teaching, documentation, or educational enablement)

### Classification Rule

- Compute the mean of all six scores
- If mean score ≥ 3.5: Classify as `ai_pedagogy_related: true`
- If mean score < 3.5: Classify as `ai_pedagogy_related: false`
- Confidence score is proportional to the mean score (0.0–1.0 scale)

### Multi-Annotator Approach

The project is designed to support:
- **Multiple LLM Judges**: Different LLM models (e.g., Llama, Qwen2, etc.) serving as independent evaluators
- **Human Annotators**: Expert human evaluators to validate and complement LLM judgments
- **Consensus Building**: Aggregation of multiple annotations for robust classification

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for model inference)
- PyTorch with CUDA support
- Hugging Face Transformers

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Jobs-Research
```

2. Install dependencies:
```bash
pip install torch transformers pandas
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

3. Authenticate with Hugging Face (required for accessing Llama models):
```bash
huggingface-cli login
```

## Usage

### Basic Usage

Evaluate job postings from a CSV file:

```bash
python LlamaJudge.py --csv_path path/to/job_postings.csv --job_posting_column job_posting
```

### Command-Line Arguments

- `--csv_path` (required): Path to the CSV file containing job postings
- `--job_posting_column` (default: `job_posting`): Column name containing the job posting text
- `--output_dir` (default: `results/JobPostings/Evaluations`): Directory to save evaluation results
- `--model_id` (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`): Hugging Face model ID for the judge
- `--max_new_tokens` (default: `512`): Maximum number of tokens to generate
- `--temperature` (default: `0.2`): Sampling temperature for generation

### Example

```bash
python LlamaJudge.py \
    --csv_path data/job_postings.csv \
    --job_posting_column description \
    --output_dir results/evaluations \
    --max_new_tokens 512 \
    --temperature 0.2
```

### CSV Format

The input CSV file should contain:
- A column with job posting text (specified by `--job_posting_column`)
- Optional: `job_id` or `id` column for tracking individual postings

Example CSV structure:
```csv
job_id,job_posting
1,"We are seeking an AI Education Specialist to develop curriculum..."
2,"Senior Machine Learning Engineer needed for product development..."
```

## Output Format

### Evaluation Results

The script generates two output files for each evaluation run:

1. **JSON File** (`job_posting_evaluations.json`): Detailed evaluation results
2. **CSV File** (`job_posting_evaluations.csv`): Tabular format for analysis

### Output Schema

Each evaluation contains:

```json
{
  "job_id": "unique_identifier",
  "job_posting": "full_job_posting_text",
  "score_pedagogy": 5,
  "score_audience": 4,
  "score_ai_relevance": 5,
  "score_responsible_ai": 3,
  "score_cross_sector_context": 5,
  "score_overall_fit": 5,
  "ai_pedagogy_related": true,
  "confidence": 0.92,
  "rationale": "Detailed explanation of the evaluation"
}
```

### Summary Statistics

The script also prints average scores and statistics:
- Average scores for each criterion
- Average confidence score
- Percentage of jobs classified as `ai_pedagogy_related: true`

## Project Structure

```
AI-Jobs-Research/
├── LlamaJudge.py          # Main evaluation script using Llama as judge
├── README.md              # This file
├── requirements.txt       # Python dependencies (if available)
└── results/               # Output directory for evaluation results
    └── JobPostings/
        └── Evaluations/   # Evaluation outputs (JSON and CSV)
```

## Evaluation Criteria Details

### 1. Pedagogical Indicators
Assesses the presence of educational activities such as:
- Curriculum development
- Tutorial creation
- Training programs
- Workshop facilitation
- Rubric design
- Educational guides and documentation

### 2. Learner/Audience Focus
Identifies the target audience:
- Students (K-12, undergraduate, graduate)
- Educators and teachers
- Professional engineers
- Software developers
- Learning communities
- Researchers

### 3. AI Relevance
Evaluates AI/ML technology mentions:
- Machine learning algorithms
- Large Language Models (LLMs)
- Generative AI (GenAI)
- Deep learning frameworks
- AI tool integration in education
- AI-assisted learning systems

### 4. Responsible AI or Governance
Checks for responsible AI practices:
- Ethics in AI education
- Fairness and bias considerations
- Transparency requirements
- Safety protocols
- Explainability in AI systems
- Responsible AI documentation

### 5. Cross-Sector Context
Assesses alignment with different sectors:
- **Academia**: University teaching, research, curriculum development
- **Edtech**: Educational technology companies, online learning platforms
- **Industry**: Corporate training, professional development, enablement
- **Government**: Public sector education, policy, training programs

### 6. Overall Fit
Holistic assessment of how well the job posting represents AI-pedagogical work combining teaching, documentation, and educational enablement.

## Future Work

### Planned Enhancements

1. **Multiple LLM Judges**: Integration of additional LLM models (Qwen2, Mistral, etc.) as independent evaluators
2. **Human Annotation Pipeline**: Tools and workflows for human annotators to review and validate classifications
3. **Inter-Annotator Agreement**: Metrics to measure agreement between LLM judges and human annotators
4. **Consensus Mechanisms**: Algorithms to aggregate multiple annotations for final classification
5. **Evaluation Dashboard**: Visualization tools for analyzing evaluation results and trends
6. **Active Learning**: Iterative refinement of the evaluation framework based on human feedback

## Contributing

This is a research project. For contributions, please:
1. Follow the existing code style and structure
2. Add appropriate documentation for new features
3. Test changes with sample data
4. Ensure compatibility with the evaluation framework

## License

[Specify license if applicable]

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{ai-jobs-research,
  title={AI Jobs Research: Classification of AI in Education Job Postings},
  author={[Your Name/Institution]},
  year={2024},
  url={[Repository URL]}
}
```

## Contact

For questions or inquiries, please contact [your contact information].

## Acknowledgments

- Meta Llama models for providing the evaluation framework
- Hugging Face for model hosting and transformers library
- The research community for feedback and contributions
