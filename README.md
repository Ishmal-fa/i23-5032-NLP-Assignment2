# i23-5032-NLP-Assignment2

**CS-4063 Natural Language Processing — Assignment 2**
FAST NUCES | Student ID: i23-5032 | Section: DS-B

> Neural NLP Pipeline on BBC Urdu Corpus — implemented entirely from scratch in PyTorch.
> No pretrained models, no HuggingFace, no Gensim.

---

## Repository Structure

```
i23-5032-NLP-Assignment2/
│
├── i23_5032_Assignment2_DS_B.ipynb   ← main notebook (all cells executed)
├── report.pdf                         ← 3-page PDF report
├── README.md                          ← this file
│
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
│
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
│
└── data/
    ├── pos_train.conll
    ├── pos_test.conll
    ├── ner_train.conll
    └── ner_test.conll
```

> **Input files required** (not committed — place in the repo root before running):
> `cleaned.txt`, `raw.txt`, `Metadata.json`

---

## Environment Setup

### Option A — Kaggle (recommended, free GPU)

1. Upload `cleaned.txt`, `raw.txt`, and `Metadata.json` as a Kaggle dataset.
2. Open the notebook on Kaggle and attach the dataset.
3. Set **Accelerator → GPU T4 x2**.
4. The path-detection cell (Cell 4) will automatically resolve file paths.
5. Click **Run All**.

### Option B — Google Colab

1. Upload `cleaned.txt`, `raw.txt`, and `Metadata.json` to the Colab `/content/` directory:
   ```python
   from google.colab import files
   files.upload()   # upload all three files
   ```
2. Open the notebook and click **Runtime → Run all**.
3. For GPU: **Runtime → Change runtime type → T4 GPU**.

### Option C — Local machine

#### Prerequisites

- Python 3.9+
- pip

#### Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn matplotlib conlleval numpy
```

> CPU-only PyTorch also works; training will be slower.

#### Place data files

```bash
# Copy your data files into the repo root
cp /path/to/cleaned.txt .
cp /path/to/raw.txt .
cp /path/to/Metadata.json .
```

#### Run the notebook

```bash
jupyter notebook i23_5032_Assignment2_DS_B.ipynb
# then: Kernel → Restart & Run All
```

Or convert to a script and run headlessly:

```bash
jupyter nbconvert --to script i23_5032_Assignment2_DS_B.ipynb
python i23_5032_Assignment2_DS_B.py
```

---

## Reproducing Each Part

All three parts are in a single notebook. Run the cells **in order from top to bottom** — later parts depend on variables defined in earlier cells.

---

### Part 1 — Word Embeddings (Cells 6–30)

**What it does:**
- Builds a 10,001-token vocabulary from `cleaned.txt`
- Assigns topic labels to all 300 articles (expanded Urdu keyword matching + title boost)
- Computes TF-IDF (10,001 × 300 matrix → `embeddings/tfidf_matrix.npy`)
- Computes PPMI co-occurrence (5,000 × 5,000, window k=5 → `embeddings/ppmi_matrix.npy`)
- Trains Skip-gram Word2Vec (d=100, k=5, K=10 negatives, 20 epochs, cosine LR)
- Evaluates via nearest neighbours, 3CosMul analogy tests, and four-condition MRR comparison

**Key outputs:**
| File | Cell | Description |
|------|------|-------------|
| `embeddings/tfidf_matrix.npy` | 9 | TF-IDF weighted term-document matrix |
| `embeddings/ppmi_matrix.npy` | 13 | PPMI co-occurrence matrix |
| `embeddings/embeddings_w2v.npy` | 21 | Averaged (V+U)/2 Skip-gram embeddings |
| `embeddings/word2idx.json` | 7 | Vocabulary index |
| `figures/w2v_loss_C3.png` | 21 | Training loss curve |
| `figures/tsne_ppmi.png` | 14 | t-SNE visualisation |

**Expected runtime:** ~25–40 min on GPU (15 epochs × 4.8M pairs); ~2–3 hr on CPU.

---

### Part 2 — Sequence Labelling: POS Tagging & NER (Cells 32–61)

**What it does:**
- Samples 500 sentences from `cleaned.txt` stratified by topic
- POS-annotates with a rule-based tagger (lexicon of 544 entries across NOUN/VERB/ADJ/ADV)
- NER-annotates with a longest-match BIO tagger (gazetteer: 54 PER, 78 LOC, 44 ORG)
- Saves CoNLL files to `data/`
- Trains a 2-layer bidirectional LSTM with dropout p=0.5 and Word2Vec initialisation:
  - **POS**: linear head + cross-entropy; frozen and fine-tuned modes compared
  - **NER**: CRF output layer + Viterbi decoding
- Evaluates token accuracy, macro-F1, confusion matrix, entity-level metrics (conlleval-style)
- Runs four ablations (A1 UniLSTM, A2 No Dropout, A3 Random Embeddings, A4 Softmax NER)

**Key outputs:**
| File | Description |
|------|-------------|
| `models/bilstm_pos.pt` | Best POS tagger checkpoint |
| `models/bilstm_ner.pt` | Best NER tagger checkpoint (CRF) |
| `data/pos_train.conll` | POS training set in CoNLL format |
| `data/pos_test.conll` | POS test set in CoNLL format |
| `data/ner_train.conll` | NER training set in CoNLL format |
| `data/ner_test.conll` | NER test set in CoNLL format |

**Expected results:**
- POS fine-tuned: **98.9% accuracy**, **0.9691 macro-F1**
- NER with CRF: **P=0.940, R=0.824, F1=0.931** (entity-level)

**Expected runtime:** ~15–30 min on GPU with early stopping (patience=5).

---

### Part 3 — Transformer Encoder for Topic Classification (Cells 63–78)

**What it does:**
- Assigns 5-class topic labels to all 300 articles (Politics, Sports, Economy, International, Health & Society)
- Encodes articles as 256-token sequences with UNK-filtered encoding
- Oversamples minority training classes to ~60 per class (Train 322 / Val 42 / Test 51)
- Implements a full Transformer encoder from scratch:
  - Sinusoidal positional encoding (fixed buffer)
  - Scaled dot-product attention with padding mask
  - Multi-head self-attention (h=4, d_model=128, d_k=d_v=32)
  - Position-wise FFN (d_ff=512, ReLU)
  - 4 stacked Pre-LayerNorm encoder blocks
  - Learned [CLS] token → MLP head (128 → 64 → 5)
- Training: AdamW (η=5e-4, weight decay=0.01), cosine warmup (100 steps), label smoothing=0.05, gradient clipping (norm=1.0), early stopping patience=12
- Trains a BiLSTM classifier (attention+mean+max pooling) on the same split for comparison
- Produces attention heatmaps from the final encoder layer for 3 correctly classified articles

**Key outputs:**
| File | Description |
|------|-------------|
| `models/transformer_cls.pt` | Best Transformer checkpoint |
| `models/bilstm_clf_best.pt` | Best BiLSTM classifier checkpoint |
| `figures/transformer_training.png` | Loss and accuracy curves |
| `figures/attention_heatmap_*.png` | Attention weight visualisations |

**Expected results:**
- BiLSTM classifier: **63.8% accuracy**, **0.4925 macro-F1**
- Transformer: **61.1% accuracy**, **0.4032 macro-F1**

**Expected runtime:** ~10–20 min on GPU (early stopping typically triggers around epoch 20–30).

---

## Restrictions (Assignment Requirements)

The following are **strictly prohibited** and are not used anywhere in this codebase:

- `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder`
- Any pretrained model weights
- HuggingFace `transformers` or `datasets`
- Gensim

---

## Hardware Used

| Component | Spec |
|-----------|------|
| GPU | Tesla T4 (Kaggle) |
| VRAM | 16 GB |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Python | 3.10 |

---

## Seed

All experiments use `SEED = 42` for full reproducibility:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## Contact

**Student ID:** i23-5032  
**Section:** DS-B  
**Course:** CS-4063 Natural Language Processing, FAST NUCES
