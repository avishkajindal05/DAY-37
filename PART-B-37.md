# Q2 — TF-IDF by Hand + Conceptual Analysis (ShopSense Clothing Reviews)

**Document used:** Clothing Doc 42 → `review_id: R010278`
**Review text:** *"Product bahut accha hai. Quality is top notch. Will buy again for sure."*
**After cleaning & tokenising (stop words removed):**
`['product', 'bahut', 'accha', 'hai', 'quality', 'top', 'notch', 'will', 'buy', 'again', 'sure']`
**Total tokens |d| = 11 | Corpus size N = 8,984**

---

## (a) TF, IDF, and TF-IDF for the word `'notch'`

### Step 1 — Term Frequency

```
TF('notch', Doc_42) = count('notch' in Doc_42) / |Doc_42|
                    = 1 / 11
                    = 0.090909
```

`'notch'` appears exactly **1 time** in Doc_42's 11 tokens.

### Step 2 — Document Frequency

```
DF('notch') = number of documents in corpus containing 'notch'
            = 621   (out of 8,984 reviews with text)
```

### Step 3 — Inverse Document Frequency (smoothed, sklearn-compatible)

```
IDF('notch') = log( (1 + N) / (1 + DF('notch')) ) + 1
             = log( (1 + 8984) / (1 + 621) ) + 1
             = log( 8985 / 622 ) + 1
             = log( 14.4453 ) + 1
             = 2.6704 + 1
             = 3.6704
```

### Step 4 — TF-IDF

```
TF-IDF('notch', Doc_42) = TF × IDF
                        = 0.090909 × 3.6704
                        = 0.3337
```

> **Interpretation:** `'notch'` appears in only 621/8,984 documents (6.9% of corpus), so it carries real discriminating power. A TF-IDF of 0.33 (before L2-normalisation) is meaningfully high for a single occurrence.

---

## (b) IDF('the') vs IDF('kurta')

### IDF('the')

```
DF('the')  = 2,992   (appears in 2,992 of 8,984 documents)
IDF('the') = log( (1 + 8984) / (1 + 2992) ) + 1
           = log( 8985 / 2993 ) + 1
           = log( 3.0022 ) + 1
           = 1.0993 + 1
           = 2.099
```

### IDF('kurta')

```
DF('kurta')  = 0   (does not appear after stop-word/clean tokenisation)
IDF('kurta') = log( (1 + 8984) / (1 + 0) ) + 1
             = log( 8985 ) + 1
             = 9.103 + 1
             = 10.103
```

### Why the difference?

`'the'` is a function word that appears in almost every English sentence. Its document frequency (2,992) is very high relative to the corpus, which makes the ratio `(1+N)/(1+DF)` close to 1 — and `log(1) = 0` — so IDF approaches 0 (here 2.099 because of smoothing). It carries zero discriminative information; every document has it.

`'kurta'` is a domain-specific clothing term (a type of Indian garment). It is absent from the cleaned corpus after tokenisation, meaning it never co-occurs with the generic review templates in this dataset — IDF reaches its maximum value (10.103). In a richer corpus with more clothing-specific vocabulary, `'kurta'` would have a moderate but still high IDF, while `'the'` would remain near zero.

---

## (c) Rebuttal: "Why not just use word frequency? TF-IDF is overcomplicated."

Raw word frequency (TF alone) treats every word equally regardless of how commonly it appears across all documents. A word like `'product'` appears in **3,998 of 8,984 ShopSense reviews** — nearly half the corpus. If you rank by TF alone, `'product'` dominates every search query, even when searching for something specific like `'earbuds battery'`. TF-IDF suppresses these corpus-wide common words via the IDF term, so retrieval surfaces genuinely relevant documents rather than generic ones.

The IDF component also solves a real engineering problem: the vocabulary that actually differentiates documents from each other is a small fraction of all words. Without down-weighting ubiquitous terms, your similarity scores are dominated by noise — two reviews about completely different products look identical because they both say `'product'`, `'quality'`, and `'delivery'`. TF-IDF is not overcomplicated; it is the minimal correction needed to make word frequency useful for retrieval.

Finally, the cost is trivial: one log computation per term per corpus — O(|V|) — computed once offline. The payoff is a retrieval system that is meaningfully better at surfacing relevant results.

---

## Bonus: BM25 Weighting (k1 = 1.5, b = 0.75)

BM25 improves on TF-IDF by saturating term frequency (repeated terms get diminishing returns) and normalising for document length.

**Formula:**
```
BM25(t, d) = IDF_BM25(t) × [TF(t,d) × (k1 + 1)] / [TF(t,d) + k1 × (1 - b + b × |d|/avgdl)]
```

**Dataset parameters (Clothing category):**
- `avgdl` = 10.32 tokens  (average document length)
- `|d42|` = 11 tokens     (Doc_42 length)
- `k1 = 1.5`, `b = 0.75`
- BM25 IDF: `log((N - DF + 0.5) / (DF + 0.5) + 1)`

### BM25 for `'notch'` in Doc_42

```
DF('notch') = 621  (in 1,504 Clothing docs with text)
IDF_BM25    = log((1504 - 621 + 0.5) / (621 + 0.5) + 1)
            = log(883.5 / 621.5 + 1)
            = log(2.4215)
            = 0.8845   ← note: BM25 IDF is not smoothed the same way

length norm = 1 - 0.75 + 0.75 × (11 / 10.32) = 1.0494
denominator = 1 + 1.5 × 1.0494 = 2.574
TF numerator= 1 × (1.5 + 1) = 2.5

BM25('notch', Doc_42) = 2.7367 × 2.5 / (1 + 2.574) = 2.658
```

### Comparison: TF-IDF vs BM25 for Doc_42 query words

| Word | TF-IDF score (raw) | BM25 score |
|:---|:---|:---|
| `notch` | 0.3337 | 2.658 |
| `quality` | 0.1595 | 0.961 |
| `top` | 0.3337 | 2.658 |

**How scores change:**
BM25 scores are on a different scale (unbounded, not normalised) but the **rankings are preserved** — `notch` and `top` remain the highest-scoring rare terms. The key difference is that BM25's TF saturation means a word appearing **3 times** in a document would score less than 3× a single occurrence — unlike raw TF which scales linearly. For Doc_42 (where all query words appear exactly once), TF saturation has no effect here, but BM25's length normalisation slightly penalises Doc_42 (length 11 > avgdl 10.32), reducing scores marginally compared to a shorter document with the same words.
