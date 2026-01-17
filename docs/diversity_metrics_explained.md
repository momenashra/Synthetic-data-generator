# Diversity Metrics - Complete Guide

## Overview

The **DiversityAnalyzer** ([diversity.py](file:///Users/momen/Desktop/easy/Synthetic-Data-Generator/quality/diversity.py)) measures lexical, semantic, and syntactic variety in synthetic reviews. High diversity indicates the generation model produces varied, non-repetitive content.

> [!NOTE]
> **Architecture Update**: All embedding operations now use the shared `EmbeddingClient` singleton from `/models`. This ensures consistent embeddings across bias, diversity, and realism analyzers.

> [!IMPORTANT]
> Diversity is crucial for creating realistic synthetic datasets. Low diversity reveals the model is "stuck" in repetitive patterns, producing cookie-cutter reviews that machine learning models can easily identify as synthetic.

---

## 🎯 Why Measure Diversity?

### Purpose
1. **Avoid Overfitting**: Training ML models on low-diversity synthetic data creates models that only recognize template patterns
2. **Realism**: Real reviews vary wildly in vocabulary, structure, and expression
3. **Coverage**: Diverse synthetic data covers more linguistic patterns and edge cases
4. **Detection Resistance**: High diversity makes synthetic reviews harder to detect

### When to Use
- ✅ After every generation batch to track quality
- ✅ When comparing different LLM models/prompts
- ✅ Before using data for downstream tasks
- ✅ During hyperparameter tuning

---

## 📊 Metrics Explained

### 1. Vocabulary Overlap

**What it does**: Measures lexical richness through token-level analysis.

**How it works**:
```python
def calculate_vocabulary_overlap(self, reviews: List[str])
```

**Metrics Computed**:

#### a) Type-Token Ratio (TTR)
**Formula**: `TTR = unique_tokens / total_tokens`

**Example**:
```python
reviews = ["Great product", "Great quality", "Product quality"]
# Tokens: great, product, great, quality, product, quality (6 total)
# Unique: great, product, quality (3 unique)
# TTR = 3/6 = 0.50
```

**Interpretation**:
- **TTR > 0.7**: Very diverse (many unique words)
- **TTR 0.4-0.7**: Moderate diversity
- **TTR < 0.4**: Repetitive (limited vocabulary)

**Limitations**:
- ❌ Sensitive to corpus size (larger corpora naturally have lower TTR)
- ❌ Doesn't account for semantic meaning

#### b) Average Unique Tokens Per Review
**Purpose**: Measures vocabulary richness at individual review level.

**Example**:
```python
review_1 = "Great amazing excellent product"  # 4 unique
review_2 = "The the the product"  # 2 unique
avg_unique = (4 + 2) / 2 = 3.0
```

**Good Range**: 8-15 unique tokens per review (for 10-30 word reviews)

#### c) Top-10 Token Concentration
**Purpose**: Detects over-reliance on common words.

**Formula**: `ratio = count(top_10_words) / total_tokens`

**Example**:
```python
# High concentration (bad)
top_10 = ['product', 'great', 'good', ...]
ratio = 0.6  # 60% of words are from top 10 ⚠️

# Low concentration (good)
ratio = 0.3  # 30% from top 10 ✅
```

**Thresholds**:
- **< 0.3**: Very diverse vocabulary
- **0.3-0.5**: Acceptable
- **> 0.5**: Over-reliance on few words

---

### 2. Semantic Similarity (Embeddings)

**What it does**: Measures meaning-level similarity using neural embeddings.

**Model Used**: Shared `EmbeddingClient` using `intfloat/e5-small-v2`
- Singleton instance (loaded once, reused everywhere)
- 384-dimensional embeddings
- Consistent with bias and realism analyzers

**How it works**:
```python
def calculate_semantic_similarity(self, reviews: List[str], sample_size: int = 100)

# Uses shared client internally:
embedding_client = self._get_embedding_client()
embeddings = embedding_client.encode(reviews)
```

**Process**:
1. Encode all reviews as 384-dimensional vectors
2. Compute pairwise cosine similarity
3. Sample pairs if n > 100 (for efficiency)
4. Calculate statistics

**Cosine Similarity Formula**:
```
similarity = (A · B) / (||A|| × ||B||)
Range: -1 (opposite) to 1 (identical)
```

**Returns**:
- `avg_similarity`: Mean similarity across all pairs
- `max_similarity`: Most similar pair (potential duplicates)
- `min_similarity`: Least similar pair
- `std_similarity`: Variance in similarity
- `embedding_diversity`: `1 - avg_similarity`

**Interpretation**:
```python
avg_similarity = 0.85  # ⚠️ Very similar (low diversity)
avg_similarity = 0.60  # Moderate
avg_similarity = 0.35  # ✅ Diverse (semantically distinct)
```

**Why This Matters**:
- Two reviews can use different words but express the same idea
- Example:
  ```
  "Amazing quality" vs "Excellent craftsmanship"
  Lexical overlap: 0%
  Semantic similarity: 0.78 (high)
  ```

**Limitations**:
- ❌ Computationally expensive (O(n²) comparisons)
- ❌ Model biases affect results
- ❌ Doesn't capture domain-specific nuances

---

### 3. Lexical Diversity (Advanced)

**What it does**: Calculates research-grade diversity metrics.

**How it works**:
```python
def calculate_lexical_diversity(self, reviews: List[str])
```

#### a) Distinct-N Scores
**Purpose**: Measures unique n-gram ratios.

**Formula**: `distinct-n = unique_n-grams / total_n-grams`

**Example**:
```python
text = "great product great quality"
# Bigrams: [great product, product great, great quality]
# Unique: 3 | Total: 3
distinct-2 = 3/3 = 1.0 ✅ Perfectly diverse

text = "great great great great"
# Bigrams: [great great, great great, great great]
# Unique: 1 | Total: 3
distinct-2 = 1/3 = 0.33 ⚠️ Highly repetitive
```

**Typical Ranges**:
- `distinct-1`: 0.3-0.6 (unigrams)
- `distinct-2`: 0.6-0.9 (bigrams)
- `distinct-3`: 0.8-0.95 (trigrams)

**Why Use Multiple N**:
- Distinct-1: Vocabulary breadth
- Distinct-2: Phrase diversity
- Distinct-3: Sentence structure variety

#### b) N-gram Entropy
**Purpose**: Measures predictability of word sequences.

**Formula**:
```
Entropy = -Σ p(ngram) × log(p(ngram))
Normalized = Entropy / log(total_unique_ngrams)
```

**Interpretation**:
- **High entropy (> 0.8)**: Unpredictable, creative language ✅
- **Low entropy (< 0.5)**: Templated, predictable patterns ⚠️

**Example**:
```python
# Low entropy (predictable)
"The product is great. The product is good. The product is nice."
# Always: "The product is [adjective]"

# High entropy (varied)
"Loved it! Quality exceeded expectations. Surprisingly comfortable design."
```

#### c) Compression Ratio Diversity
**Purpose**: Uses data compression to measure repetition.

**Methodology**:
- Concatenate all reviews
- Compress using zlib
- `compression_ratio = original_size / compressed_size`
- `diversity = 1 / compression_ratio`

**Why This Works**:
Compression algorithms exploit repetition:
- Repetitive text compresses well (high ratio, low diversity)
- Varied text compresses poorly (low ratio, high diversity)

**Example**:
```python
# Repetitive (compresses to 20% of original)
compression_ratio = 5.0
diversity_score = 1/5.0 = 0.20 ⚠️

# Diverse (compresses to 60% of original)
compression_ratio = 1.67
diversity_score = 1/1.67 = 0.60 ✅
```

---

### 4. DCScore (Diversity-Concentration Score)

**What it does**: Measures how distinct each review is from all others.

**How it works**:
```python
def calculate_dcscore(self, reviews: List[str])
```

**Algorithm**:
1. Encode reviews as normalized embeddings
2. Compute pairwise cosine similarity matrix
3. Apply row-wise softmax (converts to probability distribution)
4. Extract diagonal values (self-similarity after normalization)
5. Return mean diagonal value

**Formula**:
```python
similarity_matrix = embeddings @ embeddings.T
softmax_matrix = softmax(similarity_matrix, axis=1)
dcscore = mean(diag(softmax_matrix))
```

**Interpretation**:
- **DCScore → 1.0**: Each review is very distinct (diverse) ✅
- **DCScore → 0.0**: Reviews blur together (similar) ⚠️

**Example**:
```python
# Diverse dataset
dcscore = 0.85  # Each review stands out

# Homogeneous dataset
dcscore = 0.45  # Reviews are similar
```

**Why DCScore is Better Than Avg Similarity**:
- Accounts for the **full distribution** of similarities
- Softmax emphasizes the most similar reviews
- More sensitive to outliers and clusters

---

### 5. Cluster Inertia

**What it does**: Measures how spread out reviews are in semantic space.

**How it works**:
```python
def calculate_cluster_inertia(self, reviews: List[str], n_clusters: int = 10)
```

**Process**:
1. Encode reviews as embeddings
2. Run KMeans clustering (default k=10)
3. Compute inertia (sum of squared distances to cluster centers)

**Formula**:
```
Inertia = Σ min(||x - μⱼ||²) for all x, across all clusters j
```

**Interpretation**:
- **High inertia**: Reviews are spread out (diverse) ✅
- **Low inertia**: Reviews cluster tightly (homogeneous) ⚠️

**Example**:
```python
# Diverse (spread across semantic space)
inertia = 15000

# Clustered (groups of similar reviews)
inertia = 3000
```

**Limitations**:
- ❌ Depends on number of clusters (must tune `n_clusters`)
- ❌ Absolute values are hard to interpret
- ❌ Better for comparison (batch A vs batch B)

---

### 6. Syntactic Diversity (CR-POS)

**What it does**: Measures sentence structure variety using part-of-speech tags.

**How it works**:
```python
def calculate_syntactic_diversity(self, reviews: List[str])
```

**Process**:
1. Tokenize and POS-tag each review
   ```
   "Great product" → [('Great', 'JJ'), ('product', 'NN')]
   POS sequence: "JJ NN"
   ```
2. Concatenate all POS sequences
3. Compress using zlib
4. `diversity = 1 / compression_ratio`

**Example**:
```python
# Low diversity (always "Adjective Noun")
reviews = ["Great product", "Good quality", "Nice design"]
pos_sequences = ["JJ NN", "JJ NN", "JJ NN"]
# Compresses very well → low diversity

# High diversity (varied structures)
reviews = ["Loved it!", "Quality exceeded my expectations", "Arrived quickly"]
pos_sequences = ["VBD PRP", "NN VBD PRP$ NNS", "VBD RB"]
# Compresses poorly → high diversity
```

**Why This Matters**:
- Catches reviews that use different words but identical grammar
- Example: "Great shoes" vs "Amazing boots" (both Adj + Noun)

**Limitations**:
- ❌ Doesn't capture semantic meaning
- ❌ POS tagging errors affect results
- ❌ Language-dependent (English only)

---

### 7. TF-IDF Similarity

**What it does**: Compares reviews based on term importance.

**How it works**:
```python
def calculate_tfidf_diversity(self, reviews: List[str])
```

**TF-IDF Formula**:
```
TF-IDF(term, doc) = (count(term in doc) / total_terms) × log(total_docs / docs_containing_term)
```

**Process**:
1. Build TF-IDF matrix (reviews × words)
2. Compute pairwise cosine similarity
3. Calculate averages

**Example**:
```python
review_1 = "Great leather shoes"
review_2 = "Excellent leather boots"
# High TF-IDF for: leather (appears in both)
# Cosine similarity ≈ 0.6 (moderately similar)

review_3 = "Smartphone battery life"
# Low TF-IDF overlap with review_1
# Cosine similarity ≈ 0.1 (very different)
```

**Returns**:
- `avg_tfidf_similarity`: Mean similarity
- `std_tfidf_similarity`: Variance

**Difference from Semantic Similarity**:
- TF-IDF: Word-level matching (syntax)
- Embeddings: Meaning-level matching (semantics)

---

### 8. Semantic Deduplication

**What it does**: Identifies near-duplicate reviews for removal.

**How it works**:
```python
def semantic_deduplication(self, reviews: List[str], threshold: float = 0.9)
```

**Algorithm**:
1. Encode all reviews as embeddings
2. Compute pairwise similarities
3. For each review, mark subsequent reviews with similarity > threshold
4. Return indices of reviews to keep

**Example**:
```python
reviews = [
    "Great product",
    "Excellent item",  # 0.92 similar to #0 → discard
    "Fast shipping",   # 0.25 similar → keep
]
keep_indices = [0, 2]  # Removed review #1
```

**Use Cases**:
- Removing duplicates after generation
- Ensuring each review adds unique information
- Quality control before dataset release

**Threshold Guide**:
- **0.95+**: Only remove near-exact duplicates
- **0.90**: Recommended (removes very similar)
- **0.80**: Aggressive (may remove valid variations)

---

## 📈 Best Practices

### 1. **Use Multiple Metrics**
Don't rely on a single score:
```python
# ⚠️ Misleading
high_vocabulary_diversity = 0.8  # Good lexical diversity
low_semantic_diversity = 0.2  # But semantically repetitive!

# ✅ Comprehensive
combined_score = (vocab_diversity + semantic_diversity + syntactic_diversity) / 3
```

### 2. **Establish Baselines**
Compare against real reviews:
```python
real_reviews_diversity = analyzer.analyze(real_reviews)
synthetic_diversity = analyzer.analyze(synthetic_reviews)

if synthetic_diversity['dcscore'] >= real_reviews_diversity['dcscore'] * 0.9:
    print("✅ Synthetic diversity matches real data")
```

### 3. **Monitor Across Batches**
Track diversity over time:
```python
batch_1_diversity = 0.75
batch_2_diversity = 0.68  # Dropping
batch_3_diversity = 0.62  # ⚠️ Investigate model degradation
```

### 4. **Balance Diversity and Quality**
Maximum diversity isn't always best:
```python
# ⚠️ Too diverse (gibberish)
"Quantum potato vibes galactic shoelace"

# ✅ Diverse but coherent
"Exceeded expectations. Comfortable fit. Would recommend."
```

---

## ⚠️ Limitations & Gaps

### Current Issues

1. **Computational Cost**
   - Embedding models are slow (O(n) for encoding, O(n²) for similarity)
   - Not suitable for real-time evaluation

2. **Thresholds Are Not Validated**
   - No ground truth for "good" diversity
   - Metrics assume higher = better (not always true)

3. **Language Dependency**
   - All metrics assume English
   - POS tagging, tokenization, embeddings are English-tuned

4. **No Context Awareness**
   - High diversity in irrelevant dimensions (e.g., random product mentions) isn't helpful

### Missing Metrics

- **Topic diversity**: Do reviews cover different aspects?
- **Demographic diversity**: Persona variation
- **Temporal diversity**: Review timing patterns
- **Multi-modal diversity**: Images, videos (if applicable)

---

## 🚀 Recommended Improvements

### Priority 1: Add Topic Modeling
```python
from sklearn.decomposition import LatentDirichletAllocation

def topic_diversity(reviews, n_topics=5):
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=n_topics)
    topic_dist = lda.fit_transform(doc_term_matrix)
    # Measure entropy of topic distribution
    return -np.sum(topic_dist * np.log(topic_dist + 1e-10))
```

### Priority 2: Use Better Embedding Models
```python
# Upgrade to domain-specific model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Better quality, larger (420MB)
```

### Priority 3: Add Statistical Testing
```python
from scipy.stats import ks_2samp

# Compare diversity distributions
real_similarities = ...
synthetic_similarities = ...
statistic, p_value = ks_2samp(real_similarities, synthetic_similarities)
print(f"Distributions match: {p_value > 0.05}")
```

---

## 🎓 Summary

| Metric | Purpose | Computation Cost | Best For |
|--------|---------|------------------|----------|
| **Vocabulary Overlap** | Lexical richness | Low | Quick checks |
| **Semantic Similarity** | Meaning diversity | High | Deep analysis |
| **Lexical Diversity** | N-gram patterns | Medium | Research use |
| **DCScore** | Distinctness | High | Overall diversity |
| **Cluster Inertia** | Semantic spread | High | Batch comparison |
| **Syntactic Diversity** | Grammar variety | Medium | Structure checks |
| **TF-IDF Similarity** | Term uniqueness | Medium | Content variety |

**Best Combination**:
1. **Quick Check**: Vocabulary Overlap + Distinct-N
2. **Production**: Semantic Similarity + DCScore
3. **Research**: All metrics combined

**Key Takeaway**: Diversity metrics are **comparative tools**, not absolute measures. Always benchmark against real data from your domain.
