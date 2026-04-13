# Week 07 · Monday — NLP Foundations: TF-IDF

**Assignment:** TF-IDF from scratch on ShopSense Reviews Dataset
**Due:** Saturday 11:59 PM IST

---

## Files

| File | Description |
|:---|:---|
| `PART-A-37.py` | Q1 — TF-IDF from scratch (sparse matrix, cosine ranking, sklearn comparison, top word) |
| `PART-B-37.md` | Q2 — Hand-computed TF-IDF steps, IDF analysis, written rebuttal, BM25 bonus |

---

## Dataset



```
DAY-37/
├── shopsense_reviews.csv   
├── PART-A-37.py
├── PART-B-37.md
└── README.md
```

---

## How to Run

### Install dependencies

```bash
pip install numpy pandas scipy scikit-learn
```

### Run Q1

```bash
cd DAY-37
python3 PART-A-37.py
```

### Expected output

```
Loaded 8984 reviews
Shape: (8984, 235)  |  Vocab: 235  |  Non-zero: 92,085
Top-5 reviews for query ranked by cosine similarity
Avg L2 diff vs sklearn: ~1.34
Top word in Electronics: 'quality'  avg score: 0.0686
```

### Q2

Q2 is a written analysis — open `PART-B-37.md` to read the step-by-step hand calculation.

---

## Notes

- The ShopSense dataset contains 8,984 reviews with text (1,215 rows have `NaN` review_text, dropped on load)
- Vocabulary size is 235 unique tokens after stop-word removal and HTML cleaning
- The dataset has 348 unique review templates recycled across ~10K rows — top-5 cosine results may share the same text (expected)
- Q1 query adjusted to `'earbuds battery poor quality'` to match actual corpus vocabulary (`wireless` and `life` are not in the dataset vocabulary)
