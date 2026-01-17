# Synthetic Data Generator

A powerful tool for generating high-quality synthetic product reviews using agentic workflows (LangGraph), MCP tools, and advanced quality metrics.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Usage (CLI)

Use the `cli.py` script to run project tasks.

#### 1. Generate Reviews
Generate synthetic reviews using the agentic workflow.

```bash
# Generate 10 reviews (default)
python cli.py generate

# Generate 100 reviews
python cli.py generate --count 100
```

#### 2. Quality Report
Generate a comprehensive quality report for existing reviews.

```bash
python cli.py report --input data/generated_reviews.jsonl --output report.json
```

#### 3. Migrate Data
Generate embeddings for existing reviews (if upgrading schema).

```bash
python cli.py migrate
```

## 📚 Documentation

Detailed documentation for quality metrics is available in the `docs/` directory:

- [Quality Metrics Overview](docs/README.md)
- [Bias Metrics](docs/bias_metrics_explained.md)
- [Diversity Metrics](docs/diversity_metrics_explained.md)
- [Realism Metrics](docs/realism_metrics_explained.md)

## 🏗️ Architecture

- **Agents**: Planner, Generator, Reviewer, Comparator (LangGraph)
- **Models**: Gemini/Groq (LLM), e5-small-v2 (Embeddings)
- **Storage**: JSONL (reviews) + .npy (embeddings)
