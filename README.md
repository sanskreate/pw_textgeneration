# ✍️ Keyword-Constrained Text Generation: T5 + DMK Loss + DCKG

A research prototype for **controllable text generation**: given a few context words or sentences, generate fluent text that includes specific keywords. It combines three parts:

1. **T5 baseline:** `t5-small` fine-tuned on CommonGen to turn concept words into natural sentences.
2. **DMK (Domain-aware Model for Keywords):** a T5 model trained with a custom **keyword-weighted loss** that rewards getting keyword tokens right.
3. **DCKG (Domain-Constrained Keyword Generator):** pulls keywords out of the context with a **BERT NER** model and mixes in domain keywords.

The DCKG and DMK ideas from this prototype were carried forward into **[Freeform_TextGeneration](https://github.com/sanskreate/Freeform_TextGeneration)**, which uses Llama 3.1 for 750–2,000-word long-form generation with 100% keyword enforcement.

| | |
|---|---|
| **Base model** | `t5-small` (60M parameters, 6 encoder + 6 decoder layers) |
| **Dataset** | [allenai/common_gen](https://huggingface.co/datasets/allenai/common_gen): 5,000 concept → sentence pairs |
| **Keyword extraction** | [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) |
| **Stack** | PyTorch · Hugging Face Transformers · Datasets · SentencePiece · scikit-learn · Pandas · Streamlit · uv |

---

## 🧠 How it works

```
 Context words / sentences
           │
           ▼
 ┌─────────────────────┐   BERT NER (ORG/PER/LOC/MISC) + domain keywords
 │ DCKG keyword engine │ ─────────────────────────────────────────────┐
 └─────────────────────┘                                              │
           │                                                          ▼
           │        prompt: "generate: <context> | keywords: k1, k2, k3"
           ▼
 ┌─────────────────────┐   beam search (4 beams, up to 512 tokens)
 │  T5 (fine-tuned)    │ ──────────────────────────────────────▶ draft text
 └─────────────────────┘
           │
           ▼
 ┌─────────────────────┐   trained with DMK loss to keep keywords in
 │  DMK model (T5)     │ ──────────────────────────────────────▶ final text
 └─────────────────────┘
```

### The DMK loss
```
L = α · CrossEntropy(all tokens) + (1 − α) · CrossEntropy(keyword tokens)      with α = 0.7
```
The second term adds extra weight to the keyword positions, so the model is penalized more for dropping or garbling a required keyword than for an ordinary word.

### DCKG
- Runs NER on the context and keeps entities tagged **ORG, PER, LOC or MISC** as base keywords.
- **Diversifies** them by sampling from a list of domain keywords (default: education, learning, AI, content, students).
- Returns the combined keyword set, which conditions the generator.

---

## 📊 Data pipeline

| Step | Script | Output |
|---|---|---|
| Download CommonGen, join each sample's concepts into a context string, keep 5,000 pairs | `src/data_loader.py` | `data/common_gen.csv` (5,000 rows) |
| 90 / 10 train/validation split (`random_state=42`) | `src/preprocess.py` | `data/train.csv` (4,500) · `data/val.csv` (500) |

Example rows:

| context | text |
|---|---|
| `lean rider curve` | Two motorcycle riders are leaning deep into a curve on a racetrack. |
| `ski mountain skier` | A skier is skiing down a mountain. |

---

## 🏋️ Training

| Model | Script | Setup | Saved to |
|---|---|---|---|
| T5 baseline | `src/train_t5.py` | `t5-small`, AdamW lr 5e-5, batch 2, max length 64, 3 epochs, padding masked to −100 | `models/t5_baseline/` |
| DMK model | `src/train_gan.py` | `t5-small`, AdamW lr 5e-5, batch 8, max length 64, 3 epochs, DMK loss (α = 0.7) | `models/gan_dmk_model/` |

Both scripts use the GPU if one is available. Trained weights (`model.safetensors`) are stored with **Git LFS**, alongside the tokenizer and config files.

---

## 🧱 Project structure

```
pw_textgeneration/
├── data/
│   ├── common_gen.csv        # 5,000 CommonGen concept → sentence pairs
│   ├── train.csv             # 4,500 training rows
│   └── val.csv               # 500 validation rows
├── models/
│   ├── t5_baseline/          # Fine-tuned T5 (weights via Git LFS)
│   └── gan_dmk_model/        # DMK-loss T5 (weights via Git LFS)
├── src/
│   ├── data_loader.py        # Download + flatten CommonGen
│   ├── preprocess.py         # Train/val split
│   ├── train_t5.py           # Baseline fine-tuning
│   ├── train_gan.py          # DMK-loss training
│   ├── dckg.py               # NER-based keyword generator
│   ├── generate.py           # End-to-end generation pipeline
│   └── utils.py              # Logging + tokenization helpers
├── ui_app.py                 # Streamlit front end
├── pyproject.toml / uv.lock  # uv-managed environment (Python ≥ 3.10)
└── requirements.txt
```

---

## 🚀 Getting started

```bash
git clone https://github.com/sanskreate/pw_textgeneration.git
cd pw_textgeneration
git lfs pull                  # fetch the trained model weights

# with uv (recommended)
uv sync
# or with pip
pip install -r requirements.txt

# rebuild the data (optional, CSVs are already included)
python -m src.data_loader
python -m src.preprocess

# train
python -m src.train_t5
python -m src.train_gan

# try the keyword generator on its own
python -m src.dckg
```

---

## 🚧 Project status and known issues

This is an **experimental prototype**. The data pipeline, both training scripts and the DCKG module work on their own. The end-to-end generation path still needs wiring:

- **`src/generate.py`** imports a `TextGenerator` class that doesn't exist, and calls `generate_keywords()` without creating `DomainConstrainedKeywordGenerator` first. The fix is to load `models/gan_dmk_model` with `T5ForConditionalGeneration.from_pretrained` and call `DomainConstrainedKeywordGenerator().generate_keywords(...)`.
- **`ui_app.py`** calls `generate_text(user_input)`, but the function also needs the two model paths.
- **The DMK keyword mask** in `train_gan.py` is currently a placeholder that covers *all* target tokens, so the keyword term matches the regular loss. To make DMK work as designed, the mask should flag only the token positions of the extracted keywords.
- **Naming:** despite the `gan_` folder name, the DMK model is a T5 trained with a weighted loss. There's no adversarial discriminator yet.
- **Long form:** CommonGen targets are single sentences, so this setup produces short outputs. Long-form (750+ words) generation is handled in the follow-up project, **Freeform_TextGeneration**.

### Roadmap
- [ ] Real keyword-position masks for the DMK loss
- [ ] Wire DCKG → T5 → DMK end to end, and fix the Streamlit app
- [ ] Evaluate keyword coverage, BLEU/ROUGE and fluency on `val.csv`
- [ ] Add a discriminator for true adversarial (GAN) training

---

👩‍💻 **Author:** [Sanskriti Rai](https://www.linkedin.com/in/sanskreate/) · [sanskritirai.live](https://sanskritirai.live)
