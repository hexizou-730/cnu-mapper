# CNU Mapper

CNU Mapper is a course-description classifier for French CNU sections
(`Conseil National des Universites`). It compares four approaches:

1. A DDC-based One-vs-Rest Logistic Regression classifier.
2. A DDC-based Multi-Layer Perceptron classifier.
3. A DDC-based XLM-RoBERTa Transformer classifier.
4. A prompt-based LLM classifier via OpenRouter.

It also reports Random and Majority baselines to make the numerical evaluation
easier to interpret.

The main classical-ML pipeline predicts Dewey Decimal Classification
(DDC) codes first, then maps the predicted DDC codes to CNU sections with a
rule-based mapping table.

```text
course description -> DDC prediction -> DDC-to-CNU mapping -> CNU sections
```

The supervised learning target is DDC, because DDC codes are present in the
source thesis metadata. CNU sections are not manually annotated in the dataset;
they are derived by applying the rule-based DDC-to-CNU mapping. For this reason,
DDC metrics are the primary evaluation metrics, while mapped CNU metrics should
be interpreted as proxy downstream metrics.

## Project Structure

```text
cnu_mapper/
|-- ddc_classifier.py                  # DDC-based LogReg and MLP classifiers
|-- transformer_ddc_classifier.py      # DDC-based XLM-RoBERTa classifier
|-- llm_classifier.py                  # Prompt-based LLM classifier
|-- dewey_to_cnu.py                    # Rule-based DDC-to-CNU mapping table
|-- label_theses.py                    # Build labeled training data from raw.csv
|-- fetch_theses.py                    # Download / inspect / filter theses.fr data
|-- cnu_knowledge_base_official.json   # Official CNU reference used by DDC and LLM output
|-- rebuild_official_kb.py             # Rebuild the official CNU JSON from curated official lists
|-- DATA_PROVENANCE.md                 # Data source notes
|-- DATASET_STATISTICS.md              # Dataset size and label-distribution summary
`-- USAGE_GUIDE.md                     # Detailed usage guide
```

Large local artifacts are intentionally ignored by Git:

```text
raw.csv
theses_labeled.csv
ddc_model_logreg.pkl
ddc_model_mlp.pkl
ddc_model_xlm_roberta.pt
.env
```

Regenerate them locally when needed, or store them with Git LFS / external
storage if you want to share trained models and data.

## Methods

### 1. DDC + Logistic Regression

This is the most stable classical-ML approach.

```text
abstract -> TF-IDF -> One-vs-Rest Logistic Regression -> DDC -> CNU
```

One binary Logistic Regression model is trained for each DDC code.

### 2. DDC + MLP

This is the neural-network baseline.

```text
abstract -> TF-IDF -> MLPClassifier -> full DDC vector -> CNU
```

A single Multi-Layer Perceptron predicts the full DDC indicator vector.

### 3. DDC + XLM-RoBERTa Transformer

This is the PyTorch/Transformers approach. It uses the pretrained
`xlm-roberta-base` encoder and adds a multi-label DDC classification head.

```text
abstract -> XLM-RoBERTa -> DDC vector -> CNU
```

The model still predicts DDC first. CNU sections are produced by the same
rule-based DDC-to-CNU mapping used by LogReg and MLP.

### 4. LLM Classifier

This approach does not train a local model. It sends the course description
and the CNU knowledge base to an OpenRouter-compatible LLM and asks for CNU
section codes directly.

```text
course description -> LLM prompt -> CNU sections
```

## Quick Start

Create the Python environment:

```bash
bash setup.sh
conda activate cnu_mapper
```

Evaluate the baselines and the two DDC-based classical-ML methods:

```bash
python ddc_classifier.py --compare
```

Evaluate only the simple baselines:

```bash
python ddc_classifier.py --baselines
```

For a quick smoke test:

```bash
python ddc_classifier.py --compare --max-samples 5000
```

Train the two DDC-based models:

```bash
python ddc_classifier.py --train --model-type logreg
python ddc_classifier.py --train --model-type mlp
```

Run a small XLM-RoBERTa smoke test:

```bash
python transformer_ddc_classifier.py --eval --max-samples 5000
```

Train and save the XLM-RoBERTa model:

```bash
python transformer_ddc_classifier.py --train
```

On Apple Silicon, the script automatically uses `mps` when PyTorch supports it.

Run a prediction with a trained DDC model:

```bash
python ddc_classifier.py --model-type logreg \
  "Ce cours introduit l'apprentissage automatique, les reseaux de neurones et la fouille de donnees."
```

Expected output format:

```text
Model: DDC logreg
Decision threshold: 0.xxx
Description: ...
Predicted DDC: 004
Mapped CNU:
  27  Computer science / Informatique
```

## LLM Usage

Create a `.env` file with your OpenRouter key:

```bash
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' > .env
```

Run the LLM classifier:

```bash
python llm_classifier.py "This course covers machine learning and neural networks."
```

Interactive mode:

```bash
python llm_classifier.py
```

## Evaluation Output

The DDC comparison command prints a compact table:

```text
EVALUATION SUMMARY
Model      DDC Top-1 Acc   DDC Top-3 Acc   Mapped CNU Micro F1
--------   -------------   -------------   -------------------
Random     ...
Majority   ...
LogReg     ...
MLP        ...
```

The Transformer evaluator prints the same columns for the XLM-RoBERTa route:

```bash
python transformer_ddc_classifier.py --eval --device mps --batch-size 16 --max-length 256
```

Full 80/20 evaluation on `theses_labeled.csv`:

| Model | DDC Top-1 Acc | DDC Top-3 Acc | Mapped CNU Micro F1 |
|---|---:|---:|---:|
| Random | 1.12% | 1.12% | 0.0257 |
| Majority | 12.29% | 12.29% | 0.1247 |
| LogReg | 62.72% | 89.13% | 0.6650 |
| MLP | 52.21% | 77.27% | 0.5663 |
| XLM-R | 60.95% | 87.66% | 0.6542 |

The XLM-RoBERTa row was produced with one full epoch on Apple Silicon MPS,
`batch-size=16`, and `max-length=256`. The training stage took about 10.98
hours and ended with training loss `0.0298`. It is clearly stronger than the
MLP baseline, but the TF-IDF + Logistic Regression baseline remains the best
method in the current full evaluation.

Evaluation setup:

- Baseline train split: 265,286 samples
- Fit split: 212,228 samples
- Validation split: 53,058 samples
- Test split: 66,322 samples
- LogReg calibrated threshold: 0.875
- MLP calibrated threshold: 0.950
- XLM-R calibrated threshold: 0.325
- XLM-R validation DDC micro-F1: 0.6186

The Random baseline uniformly samples one DDC code from the training label
space. The Majority baseline always predicts the most frequent DDC code in the
training split. Both are mapped through the same DDC-to-CNU table as LogReg and
MLP.

These metrics are used because DDC is the real supervised target and CNU is a
mapped output:

- `DDC Top-1 Acc`: whether the highest-scoring DDC candidate is correct.
- `DDC Top-3 Acc`: whether the true DDC appears among the three highest-scoring
  candidates.
- `Mapped CNU Micro F1`: proxy downstream score after applying the
  DDC-to-CNU mapping to predicted and source DDC codes.

## Data Workflow

If `raw.csv` is not already available locally, download it:

```bash
python fetch_theses.py --download
```

Build the labeled dataset:

```bash
python label_theses.py
```

The current labeling script keeps French abstracts with length between 100 and
4000 characters and writes:

```text
theses_labeled.csv
```

`rebuild_official_kb.py` is separate from the thesis CSV workflow. It does not
download data, scrape websites, or read `raw.csv`. Instead, it contains manually
curated official CNU section lists plus source URLs for verification, and it
rewrites `cnu_knowledge_base_official.json` from those local lists. In short:
`raw.csv` provides thesis abstracts and DDC metadata for training, while
`cnu_knowledge_base_official.json` provides the official meaning of CNU section
codes.

## Notes for GitHub

Do not commit API keys, raw data, generated datasets, or trained model files.
The included `.gitignore` excludes these artifacts by default.

Recommended files to commit:

```text
*.py
*.md
*.json
requirements.txt
setup.sh
.gitignore
```

Recommended files to keep local or publish separately:

```text
raw.csv
theses_labeled.csv
ddc_model_logreg.pkl
ddc_model_mlp.pkl
ddc_model_xlm_roberta.pt
.env
```

## Documentation

See `USAGE_GUIDE.md` for detailed commands and workflow explanations.
See `DATA_PROVENANCE.md` for data-source notes.
See `DATASET_STATISTICS.md` for dataset-distribution statistics.
