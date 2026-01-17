# Bias Detection Metrics - Complete Guide

## Overview

The **BiasAnalyzer** ([bias.py](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/quality/bias.py)) is responsible for detecting artificial patterns, skewed distributions, and unrealistic uniformity in synthetic reviews that would indicate bias in the generation process.

> [!IMPORTANT]
> Bias detection is critical for ensuring your synthetic data mirrors real-world review distributions. High bias indicates the model is systematically favoring certain patterns, ratings, or linguistic structures.

---

## 🎯 Why Use Bias Detection?

### Purpose
1. **Quality Assurance**: Ensure generated reviews aren't artificially skewed toward positive/negative sentiment
2. **Distribution Matching**: Verify synthetic data matches expected real-world rating distributions
3. **Pattern Detection**: Identify repetitive phrases that reveal algorithmic generation
4. **Authenticity**: Catch unnatural uniformity in review lengths and structures

### When to Use
- ✅ After every batch generation to validate quality
- ✅ Before using synthetic data for training ML models
- ✅ When comparing different generation approaches
- ✅ During A/B testing of generation prompts

---

## 📊 Metrics Explained

### 1. Rating Distribution Analysis

**What it does**: Compares the actual distribution of star ratings (1-5) against an expected distribution.

**How it works**:
```python
def analyze_rating_distribution(self, ratings: List[int], expected_dist: Dict[int, float])
```

**Mathematical Foundation**:
- Calculates actual frequency: `actual_dist[r] = count(r) / total_reviews`
- Computes absolute deviation: `|actual - expected|` for each rating
- Total deviation: `Σ|actual - expected|` across all ratings
- **Threshold**: `total_deviation > 0.15` indicates significant bias

**Example**:
```python
expected = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.30}
actual   = {1: 0.03, 2: 0.08, 3: 0.22, 4: 0.40, 5: 0.27}
# Deviation = |0.03-0.05| + |0.08-0.10| + ... = 0.08 ✅ Not biased
```

**Returns**:
- `actual_distribution`: Observed rating frequencies
- `expected_distribution`: Target frequencies from config
- `deviations`: Per-rating deviations
- `total_deviation`: Sum of all deviations
- `is_biased`: Boolean flag (deviation > 0.15)

**Why 0.15 threshold?**
- Allows for natural variation (±15% total deviation)
- Too strict (< 0.10) flags minor statistical noise
- Too lenient (> 0.20) misses significant skew

> [!TIP]
> For production use, consider using **Chi-square test** instead of simple deviation sum for statistical rigor.

---

### 2. Sentiment-Rating Consistency

**What it does**: Detects contradictions between semantic sentiment and star ratings using **embedding-based analysis**.

> [!NOTE]
> **Updated Implementation**: This metric now uses the shared `EmbeddingClient` for semantic sentiment analysis instead of simple word lists. This provides more accurate detection by understanding context, not just word presence.

**How it works**:
```python
def analyze_sentiment_consistency(self, reviews: List[Dict])
```

**New Semantic Approach**:
1. **Sentiment Anchors**: Phrases representing positive/negative sentiment are embedded
2. **Centroid Computation**: Average embedding for positive vs negative sentiments
3. **Cosine Similarity**: Each review is compared to both sentiment centroids
4. **Inconsistency Detection**: Mismatch when rating direction conflicts with semantic direction

**Sentiment Anchor Phrases**:
```python
positive_anchors = [
    "excellent product I love it",
    "highly recommend great quality",
    "fantastic amazing experience",
    "best purchase very satisfied"
]
negative_anchors = [
    "terrible product I hate it",
    "do not buy poor quality",
    "awful horrible experience",
    "worst purchase very disappointed"
]
```

**Example**:
```python
# Using EmbeddingClient (all-MiniLM-L6-v2)
from models import EmbeddingClient
ec = EmbeddingClient()

review_embedding = ec.encode_single("Battery died fast, waste of money")
pos_sim = cosine(review_embedding, pos_centroid)  # 0.35
neg_sim = cosine(review_embedding, neg_centroid)  # 0.72
# → Negative sentiment detected

# Inconsistency if rating = 5 but neg_sim > pos_sim
```

**Returns**:
- `total_reviews`: Number of reviews analyzed
- `inconsistencies_found`: Count of semantic mismatches
- `inconsistency_rate`: Ratio of inconsistent reviews
- `is_consistent`: `True` if rate < 0.1 (10%)
- `examples`: First 5 inconsistent reviews with similarity scores

**Advantages over Word Lists**:
- ✅ **Context-aware**: "not terrible" correctly identified as positive
- ✅ **Handles nuance**: Captures sentiment from sentence structure
- ✅ **Domain-agnostic**: Works across product categories
- ✅ **Consistent embeddings**: Uses shared `EmbeddingClient` singleton

**Threshold**: `sim_diff > 0.05` required for meaningful inconsistency detection

---

### 3. Repetitive Pattern Detection

**What it does**: Identifies overused phrases and duplicate reviews.

**How it works**:
```python
def detect_repetitive_patterns(self, reviews: List[str])
```

**Methodology**:
1. **N-gram Extraction**: Extracts 3-grams and 4-grams from all reviews
2. **Frequency Analysis**: Counts how many reviews contain each phrase
3. **Threshold**: Phrases appearing in >15% of reviews are flagged
4. **Duplicate Detection**: Counts identical reviews

**Example**:
```python
# Repetitive phrase detected
"exceeded my expectations" appears in 50 out of 200 reviews (25%) ⚠️

# Duplicate reviews
5 identical reviews found → duplicate_rate = 0.025 (2.5%)
```

**Returns**:
- `total_reviews`: Total number analyzed
- `unique_reviews`: Count of distinct reviews
- `duplicate_rate`: 1 - (unique / total)
- `repetitive_phrases`: List of overused phrases with frequency
- `has_repetition_issues`: `True` if >5 repetitive phrases OR duplicate_rate > 0.05

**Thresholds**:
- **Phrase frequency**: 15% (appears in 15+ reviews out of 100)
- **Duplicate rate**: 5% (exact copies)
- **Flag threshold**: >5 repetitive phrases

**Why these thresholds?**
- Real reviews naturally share common phrases (e.g., "arrived on time")
- 15% allows natural overlap while catching AI repetition
- 5% duplicates accounts for legitimate similar experiences

> [!WARNING]
> This metric uses simple string matching. Two reviews with minor variations ("great product" vs "great product!") are treated as different.

**Improvements**:
- Use **fuzzy matching** (edit distance) for near-duplicates
- Apply **semantic similarity** instead of exact string matching
- Weight phrases by **TF-IDF** to ignore common terms

---

### 4. Length Distribution Analysis

**What it does**: Detects unnatural uniformity in review word counts.

**How it works**:
```python
def analyze_length_distribution(self, reviews: List[str])
```

**Statistical Approach**:
- Calculates mean and standard deviation of word counts
- **Formula**: `std = sqrt(Σ(length - mean)² / n)`
- **Red flag**: `std < 10` indicates suspiciously uniform lengths

**Example**:
```python
# Natural (diverse lengths)
lengths = [12, 25, 8, 45, 18, 33, 7, 50]
std = 15.2 ✅ Diverse

# Suspicious (too uniform)
lengths = [20, 21, 19, 22, 20, 21, 19]
std = 1.1 ⚠️ Unnaturally uniform
```

**Returns**:
- `avg_length`: Mean word count
- `std_length`: Standard deviation
- `min_length`: Shortest review
- `max_length`: Longest review
- `is_too_uniform`: `True` if std < 10

**Why std < 10 is suspicious?**
- Real reviews vary: some are brief ("Love it!"), others detailed (50+ words)
- LLMs tend to generate similar-length outputs when given fixed prompts
- Std < 10 suggests the model is following a template

**Limitations**:
- ❌ Threshold is arbitrary (depends on product category)
- ❌ Doesn't account for intentional length constraints in prompts
- ❌ Simple metric (doesn't check structure diversity)

**Better Alternatives**:
- Use **Kolmogorov-Smirnov test** to compare against real review length distribution
- Check **sentence count distribution**, not just word count
- Analyze **character count** and **punctuation usage** patterns

---

## 🔍 Overall Bias Assessment

The `analyze()` method combines all metrics:

```python
has_bias = (
    rating_distribution['is_biased'] OR
    NOT sentiment_consistency['is_consistent'] OR
    repetitive_patterns['has_repetition_issues'] OR
    length_distribution['is_too_uniform']
)
```

**Interpretation**:
- ✅ `False`: All tests passed → High-quality synthetic data
- ⚠️ `True`: At least one test failed → Review generation settings

---

## 📈 Best Practices

### 1. **Set Realistic Expected Distributions**
```yaml
# ❌ Unrealistic
expected_ratings: {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

# ✅ Realistic (matches real e-commerce data)
expected_ratings: {1: 0.05, 2: 0.08, 3: 0.15, 4: 0.35, 5: 0.37}
```

### 2. **Monitor Trends Over Time**
Track bias metrics across generation batches to catch drift:
```python
batch_1_bias = 0.08  # Good
batch_2_bias = 0.13  # Acceptable
batch_3_bias = 0.19  # ⚠️ Investigate prompt changes
```

### 3. **Combine with Diversity Metrics**
Bias detection alone isn't enough:
- Low bias + high diversity = ✅ Excellent
- Low bias + low diversity = ⚠️ Repetitive but balanced
- High bias + high diversity = ⚠️ Skewed distribution

### 4. **Use Domain-Specific Tuning**
Different product categories have different patterns:
- **Electronics**: Expect technical terms, mixed ratings
- **Fashion**: More positive skew, subjective language
- **Food**: Strong sentiment, sensory descriptions

---

## ⚠️ Limitations & Gaps

### Current Limitations

1. **Simplistic Sentiment Analysis**
   - Word lists miss context and nuance
   - No handling of sarcasm or irony
   - Domain-agnostic (ignores product-specific terms)

2. **No Temporal Bias Detection**
   - Doesn't check if reviews change over time
   - Misses sudden topic shifts

3. **No Cross-Metric Correlation**
   - Doesn't detect relationships between bias types
   - Example: High rating bias + low length variance might indicate template generation

4. **Arbitrary Thresholds**
   - 0.15 deviation, 15% phrase frequency, std < 10
   - Not validated against real datasets

### Missing Metrics

- **Emoji usage bias**: Real reviews use emojis in specific patterns
- **Capitalization patterns**: SHOUTING vs normal case
- **Punctuation abuse**: Multiple exclamations (!!!)
- **Named entity distribution**: Brand names, locations
- **Review timestamp patterns**: Clustering detection

---

## 🚀 Recommended Improvements

### Priority 1: Statistical Rigor
```python
from scipy.stats import chisquare

def improved_rating_analysis(actual, expected):
    chi2, p_value = chisquare(actual, expected)
    return {'is_biased': p_value < 0.05}  # 95% confidence
```

### Priority 2: Semantic Sentiment
```python
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def improved_sentiment_check(text, rating):
    result = sentiment_analyzer(text)[0]
    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
    expected_score = (rating - 3) / 2  # Map 1-5 to -1 to 1
    return abs(score - expected_score) < 0.3
```

### Priority 3: Semantic Deduplication
```python
from sentence_transformers import SentenceTransformer

def semantic_repetition(reviews):
    embeddings = model.encode(reviews)
    similarities = cosine_similarity(embeddings)
    # Flag if >10% of pairs have similarity > 0.9
```

---

## 📚 When NOT to Use These Metrics

- ❌ **Single review evaluation**: Metrics need statistical samples (n > 50)
- ❌ **Cross-domain comparison**: Thresholds vary by product type
- ❌ **Real-time generation**: Computationally expensive for on-the-fly checks
- ❌ **Non-English reviews**: Sentiment lexicons are English-specific

---

## 🎓 Summary

| Metric | Purpose | Threshold | Best For |
|--------|---------|-----------|----------|
| **Rating Distribution** | Detect skewed ratings | deviation < 0.15 | Ensuring balanced sentiment |
| **Sentiment Consistency** | Catch rating-text mismatch | inconsistency < 10% | Quality control |
| **Repetitive Patterns** | Find AI fingerprints | phrase_freq < 15% | Authenticity verification |
| **Length Uniformity** | Detect templating | std > 10 words | Structural diversity |

**Overall Assessment**: These metrics provide a **solid foundation** for bias detection but would benefit from:
- Machine learning-based sentiment analysis
- Statistical hypothesis testing
- Semantic similarity measures
- Domain-specific tuning

For production systems, consider upgrading to transformer-based approaches and validating thresholds against your specific product domain's real review data.
