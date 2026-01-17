# Quality Evaluation Metrics - Complete Overview

## 📋 Quick Reference

This is your comprehensive guide to the quality evaluation system for synthetic review generation. The system consists of three complementary analyzers:

| Analyzer | Purpose | Key Metrics | Primary Use |
|----------|---------|-------------|-------------|
| **[BiasAnalyzer](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/quality/bias.py)** | Detect artificial patterns | Rating distribution, sentiment consistency, repetitive phrases | Ensuring balanced, non-skewed generation |
| **[DiversityAnalyzer](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/quality/diversity.py)** | Measure variety | Vocabulary richness, semantic similarity, syntactic diversity | Preventing template-based repetition |
| **[RealismAnalyzer](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/quality/realism.py)** | Verify authenticity | Aspect coverage, readability, AI patterns, personal pronouns | Ensuring human-like authenticity |

---

## 🏗️ Architecture Overview

> [!NOTE]
> **Updated Architecture**: All embedding operations now use a shared `EmbeddingClient` singleton. Review embeddings are stored separately from review JSON for efficiency.

### Centralized Components

| Component | Path | Purpose |
|-----------|------|---------|
| **[EmbeddingClient](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/models/embedding_client.py)** | `/models/embedding_client.py` | Singleton embedding model (all-MiniLM-L6-v2) |
| **[ReviewStorage](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/models/review_storage.py)** | `/models/review_storage.py` | Incremental JSONL + .npy embedding storage |

### File Structure
```
data/
├── generated_reviews.jsonl   # Review text + metadata (no embeddings)
└── review_embeddings.npy     # (N, 384) numpy embedding matrix
```

### Usage
```python
# Shared embedding client
from models import EmbeddingClient
ec = EmbeddingClient()  # Singleton, loads model once
embedding = ec.encode_single("Great product!")

# Review storage with embeddings
from models.review_storage import get_review_storage
storage = get_review_storage()
reviews = storage.load_all_reviews()
embeddings = storage.load_all_embeddings()  # Shape: (N, 384)
```

---

## 📚 Detailed Documentation

### 1. [Bias Detection Metrics](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/bias_metrics_explained.md)

**What it does**: Identifies systematic skews and artificial uniformity in generated reviews.

**Key Metrics**:
- ✅ **Rating Distribution Analysis**: Compares actual vs expected rating frequencies
- ✅ **Sentiment-Rating Consistency**: Detects contradictions between star ratings and review text
- ✅ **Repetitive Pattern Detection**: Identifies overused phrases and duplicate content
- ✅ **Length Distribution**: Flags unnatural uniformity in review word counts

**When to use**:
- After every generation batch
- When validating rating distribution targets
- Before using data for ML training

**Read the full guide**: [bias_metrics_explained.md](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/bias_metrics_explained.md)

---

### 2. [Diversity Metrics](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/diversity_metrics_explained.md)

**What it does**: Measures lexical, semantic, and syntactic variety to ensure non-repetitive content.

**Key Metrics**:
- ✅ **Vocabulary Overlap**: Type-Token Ratio, unique tokens, word concentration
- ✅ **Semantic Similarity**: Embedding-based meaning comparison (using SentenceTransformers)
- ✅ **Lexical Diversity**: Distinct-N scores, n-gram entropy, compression ratio
- ✅ **DCScore**: Diversity-concentration metric for distinctness
- ✅ **Cluster Inertia**: Semantic space spread using K-Means
- ✅ **Syntactic Diversity**: POS-tag compression for grammar variety
- ✅ **TF-IDF Similarity**: Term importance-based comparison

**When to use**:
- Comparing different generation models/prompts
- Ensuring coverage of linguistic patterns
- Quality gates before dataset release

**Read the full guide**: [diversity_metrics_explained.md](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/diversity_metrics_explained.md)

---

### 3. [Realism Metrics](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/realism_metrics_explained.md)

**What it does**: Validates that reviews sound like authentic human-written content.

**Key Metrics**:
- ✅ **Aspect Coverage**: Checks for product-specific feature mentions
- ✅ **Readability Analysis**: Flesch Reading Ease score (targets 50-90 range)
- ✅ **AI Pattern Detection**: Identifies telltale LLM phrases ("delve into", "in conclusion")
- ✅ **Personal Pronoun Usage**: Verifies first-person perspective ("I", "my")

**When to use**:
- Final validation before production deployment
- A/B testing generation approaches
- Anti-AI-detection verification

**Read the full guide**: [realism_metrics_explained.md](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/realism_metrics_explained.md)

---

## 🎯 Complete Workflow

### Step 1: Generation
```python
from agents import PlannerAgent, GeneratorAgent

planner = PlannerAgent(config)
generator = GeneratorAgent(config, provider='groq')

plans = [planner.create_plan(rating=r) for r in ratings]
reviews = [generator.execute_plan(plan) for plan in plans]
```

### Step 2: Bias Check
```python
from quality.bias import BiasAnalyzer

bias_analyzer = BiasAnalyzer()
bias_results = bias_analyzer.analyze(
    reviews=[{'rating': r, 'text': t} for r, t in zip(ratings, reviews)],
    expected_rating_dist={1: 0.05, 2: 0.08, 3: 0.15, 4: 0.35, 5: 0.37}
)

if bias_results['overall_bias_detected']:
    print("⚠️ Bias detected - review generation settings")
```

### Step 3: Diversity Check
```python
from quality.diversity import DiversityAnalyzer

diversity_analyzer = DiversityAnalyzer()
diversity_results = diversity_analyzer.analyze(reviews)

print(f"Semantic Diversity: {diversity_results['semantic_similarity']['embedding_diversity']}")
print(f"DCScore: {diversity_results['dcscore']}")
print(f"Distinct-2: {diversity_results['lexical_diversity']['distinct_2']}")
```

### Step 4: Realism Check
```python
from quality.realism import RealismAnalyzer

realism_analyzer = RealismAnalyzer(
    product_context={
        'category': 'footwear',
        'aspects': ['comfort', 'fit', 'durability', 'style']
    }
)

realism_results = realism_analyzer.analyze(reviews)

if not realism_results['overall_realistic']:
    print("⚠️ Reviews lack realism markers")
```

### Step 5: Decision Gate
```python
def quality_gate(bias, diversity, realism):
    """All-or-nothing quality gate."""
    return (
        not bias['overall_bias_detected'] and
        diversity['dcscore'] > 0.6 and
        diversity['lexical_diversity']['distinct_2'] > 0.7 and
        realism['overall_realistic']
    )

if quality_gate(bias_results, diversity_results, realism_results):
    print("✅ Synthetic data passed quality checks")
    # Proceed with deployment
else:
    print("❌ Quality gate failed - regenerate or tune parameters")
```

---

## 🔧 Configuration Best Practices

### 1. Tune Thresholds to Your Domain

Different product categories need different validation:

```python
# Electronics (technical)
config = {
    'bias': {
        'max_deviation': 0.15,
        'max_inconsistency_rate': 0.10
    },
    'diversity': {
        'min_dcscore': 0.65,
        'min_distinct_2': 0.75
    },
    'realism': {
        'min_aspect_coverage': 0.70,  # Technical products → more specific
        'readability_range': (50, 80)  # Allow complex terms
    }
}

# Fashion (subjective)
config = {
    'bias': {
        'max_deviation': 0.20,  # Allow rating skew (fashion is subjective)
        'max_inconsistency_rate': 0.15
    },
    'diversity': {
        'min_dcscore': 0.60,
        'min_distinct_2': 0.70
    },
    'realism': {
        'min_aspect_coverage': 0.60,  # More descriptive language
        'readability_range': (60, 90)  # Casual tone
    }
}
```

### 2. Establish Baselines from Real Data

```python
from quality.bias import BiasAnalyzer
from quality.diversity import DiversityAnalyzer
from quality.realism import RealismAnalyzer

# Analyze your real reviews
real_reviews = load_real_reviews()  # Your existing dataset

bias_baseline = BiasAnalyzer().analyze(real_reviews, ...)
diversity_baseline = DiversityAnalyzer().analyze([r['text'] for r in real_reviews])
realism_baseline = RealismAnalyzer(...).analyze([r['text'] for r in real_reviews])

# Set thresholds based on real data
THRESHOLD_DCSCORE = diversity_baseline['dcscore'] * 0.9  # Within 10% of real
THRESHOLD_ASPECT_COVERAGE = realism_baseline['aspect_coverage']['coverage_rate'] * 0.85
```

### 3. Create Monitoring Dashboards

Track metrics over time:

```python
import pandas as pd
import matplotlib.pyplot as plt

metrics_history = []

for batch_id in range(num_batches):
    reviews = generate_batch(...)
    
    metrics = {
        'batch_id': batch_id,
        'bias_score': analyze_bias(...)['total_deviation'],
        'dcscore': analyze_diversity(...)['dcscore'],
        'aspect_coverage': analyze_realism(...)['aspect_coverage']['coverage_rate'],
        'timestamp': datetime.now()
    }
    metrics_history.append(metrics)

df = pd.DataFrame(metrics_history)

# Plot trends
df.plot(x='batch_id', y=['bias_score', 'dcscore', 'aspect_coverage'])
plt.axhline(y=BIAS_THRESHOLD, color='r', linestyle='--', label='Bias Threshold')
plt.show()
```

---

## ⚠️ Common Pitfalls

### 1. **Over-Optimizing for Metrics**

❌ **Bad**:
```python
# Artificially inflating diversity by adding random words
review = base_review + random.choice(filler_words)
```

✅ **Good**:
```python
# Improve diversity through better prompting
prompt = f"Write a {persona} review focusing on {random.choice(aspects)}"
```

### 2. **Ignoring Inter-Metric Correlations**

❌ **Bad**:
```python
# Treating metrics independently
if diversity_score > threshold:
    return "PASS"
```

✅ **Good**:
```python
# Check for contradictions
if diversity_score > 0.9 and bias_score < 0.05:
    # High diversity + low bias is GOOD
elif diversity_score > 0.9 and bias_score > 0.2:
    # High diversity + high bias → random generation (BAD)
    return "FAIL"
```

### 3. **Not Validating Against Humans**

Always sample-check with human reviewers:

```python
# Blind test: Can humans distinguish synthetic from real?
test_set = real_reviews[:50] + synthetic_reviews[:50]
random.shuffle(test_set)

human_labels = get_human_annotations(test_set)
detection_accuracy = calculate_accuracy(human_labels)

# Target: <60% detection (chance level)
if detection_accuracy > 0.7:
    print("⚠️ Humans can easily detect synthetic reviews")
```

---

## 🚀 Recommended Improvements

### Near-Term (Easy Wins)

1. **Add Statistical Rigor**
   ```python
   from scipy.stats import chisquare, ks_2samp
   
   # Replace simple deviation with chi-square test
   chi2, p_value = chisquare(actual_dist, expected_dist)
   is_biased = p_value < 0.05
   ```

2. **Semantic Aspect Matching**
   ```python
   from sentence_transformers import util
   
   # Replace substring matching with embedding similarity
   aspect_sim = util.cos_sim(review_embedding, aspect_embeddings)
   ```

3. **Better AI Detection**
   ```python
   from transformers import pipeline
   
   detector = pipeline("text-classification", 
                      model="Hello-SimpleAI/chatgpt-detector-roberta")
   ```

### Long-Term (Research Upgrades)

1. **Multi-Model Ensemble**
   - Combine multiple diversity metrics using PCA or ICA
   - Weight metrics by correlation with human judgment

2. **Adversarial Testing**
   - Train a discriminator to detect synthetic reviews
   - Use feedback to improve generation

3. **Domain Adaptation**
   - Fine-tune metrics on domain-specific data
   - Learn thresholds via supervised learning

---

## 📊 Metric Summary Table

| Category | Metric Name | Range | Good Value | Compute Cost | Best For |
|----------|-------------|-------|------------|--------------|----------|
| **Bias** | Rating Deviation | 0-1 | <0.15 | Low | Distribution check |
| **Bias** | Inconsistency Rate | 0-1 | <0.10 | Low | Sentiment validation |
| **Bias** | Duplicate Rate | 0-1 | <0.05 | Low | Repetition check |
| **Diversity** | Type-Token Ratio | 0-1 | >0.5 | Low | Vocabulary breadth |
| **Diversity** | Embedding Diversity | 0-1 | >0.6 | High | Semantic variety |
| **Diversity** | Distinct-2 | 0-1 | >0.7 | Medium | Phrase diversity |
| **Diversity** | DCScore | 0-1 | >0.6 | High | Overall distinctness |
| **Realism** | Aspect Coverage | 0-1 | >0.6 | Low | Domain relevance |
| **Realism** | Flesch Score | 0-100 | 50-90 | Low | Readability |
| **Realism** | AI Pattern Rate | 0-1 | <0.05 | Low | Detection resistance |
| **Realism** | Pronoun Rate | 0-1 | >0.5 | Low | Personal perspective |

---

## 🎓 Final Recommendations

### For Quick Validation (Development)
Use lightweight metrics:
- ✅ Rating distribution
- ✅ Type-Token Ratio
- ✅ Aspect coverage
- ✅ AI pattern detection

### For Production Quality Gates
Add semantic metrics:
- ✅ All bias metrics
- ✅ Semantic similarity (embeddings)
- ✅ DCScore
- ✅ All realism metrics
- ✅ Human sample validation

### For Research & Benchmarking
Use complete suite:
- ✅ All metrics from all three analyzers
- ✅ Statistical significance tests
- ✅ Cross-validation with real data
- ✅ Adversarial testing

---

## 📞 Getting Help

For detailed explanations, see:
- **Bias**: [bias_metrics_explained.md](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/bias_metrics_explained.md)
- **Diversity**: [diversity_metrics_explained.md](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/diversity_metrics_explained.md)
- **Realism**: [realism_metrics_explained.md](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/docs/realism_metrics_explained.md)

For code examples:
- Check the [quality](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/quality) directory for implementations
- See unit tests (if available) for usage patterns

---

**Remember**: Metrics are tools, not goals. The ultimate test is: **Can humans distinguish your synthetic reviews from real ones?** If yes, iterate. If no, you've succeeded.
