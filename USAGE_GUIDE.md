# Usage Guide

This guide explains how to set up the project, rebuild the data, train the
models, evaluate them, and run predictions.

## 1. Requirements

- macOS or Linux
- Miniconda or Anaconda
- Python 3.11 environment created by `setup.sh`
- OpenRouter API key only if you want to use the LLM classifier

## 2. Setup

From the project directory:

```bash
cd /path/to/cnu_mapper_copy
bash setup.sh
conda activate cnu_mapper
```

The setup script creates a conda environment named `cnu_mapper` and installs
the packages listed in `requirements.txt`.

## 3. Repository Files

Main files:

```text
ddc_classifier.py
llm_classifier.py
dewey_to_cnu.py
label_theses.py
fetch_theses.py
cnu_knowledge_base_official.json
```

`ddc_classifier.py` and `llm_classifier.py` read the official CNU knowledge
base for section codes and French names. English display names are maintained
in `dewey_to_cnu.py`.

Generated local files:

```text
raw.csv
theses_labeled.csv
ddc_model_logreg.pkl
ddc_model_mlp.pkl
```

These generated files are not meant to be committed to GitHub by default.

## 4. Data Preparation

### 4.1 Download the raw theses dataset

```bash
python fetch_theses.py --download
```

This downloads `raw.csv` from the official French theses dataset.

### 4.2 Inspect the raw CSV schema

```bash
python fetch_theses.py --inspect
```

This prints the detected delimiter, column names, and a sample row.

### 4.3 Build the labeled training dataset

```bash
python label_theses.py
```

The script:

1. Reads `raw.csv`.
2. Keeps French abstracts between 100 and 4000 characters.
3. Extracts DDC codes from `oai_set_specs`.
4. Maps DDC codes to CNU sections using `dewey_to_cnu.py`.
5. Writes `theses_labeled.csv`.

Output columns:

```text
nnt, discipline, resume, ddc, cnu_codes, n_labels
```

## 5. DDC-Based Classifiers

Both classical-ML methods use this high-level pipeline:

```text
abstract -> TF-IDF -> DDC prediction -> DDC-to-CNU mapping -> CNU sections
```

### 5.1 Logistic Regression

This trains one binary classifier per DDC code:

```text
One-vs-Rest Logistic Regression
```

Evaluate:

```bash
python ddc_classifier.py --eval --model-type logreg
```

Train and save the final model:

```bash
python ddc_classifier.py --train --model-type logreg
```

The trained model is saved as:

```text
ddc_model_logreg.pkl
```

Predict one course description:

```bash
python ddc_classifier.py --model-type logreg \
  "Ce cours introduit l'apprentissage automatique, les algorithmes et les reseaux de neurones."
```

### 5.2 Multi-Layer Perceptron

This trains one neural network that predicts the full DDC vector:

```text
MLPClassifier
```

Evaluate:

```bash
python ddc_classifier.py --eval --model-type mlp
```

Train and save the final model:

```bash
python ddc_classifier.py --train --model-type mlp
```

The trained model is saved as:

```text
ddc_model_mlp.pkl
```

Predict one course description:

```bash
python ddc_classifier.py --model-type mlp \
  "Ce cours porte sur la chimie organique, les reactions chimiques et la catalyse."
```

### 5.3 Compare Logistic Regression and MLP

Run both evaluations and print one compact table:

```bash
python ddc_classifier.py --compare
```

For a quick smoke test:

```bash
python ddc_classifier.py --compare --max-samples 5000
```

Expected final output:

```text
EVALUATION SUMMARY
Model    DDC Micro F1   CNU Micro F1   CNU Subset Accuracy
------   ------------   ------------   -------------------
LogReg   ...
MLP      ...
```

Important: `--eval` and `--compare` train temporary models for fair evaluation.
They do not load the final `.pkl` files created by `--train`.

## 6. LLM Classifier

The LLM classifier directly predicts CNU sections from a course description.

```text
course description -> prompt -> CNU sections
```

### 6.1 Configure OpenRouter

Create a `.env` file:

```bash
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' > .env
```

### 6.2 One-shot prediction

```bash
python llm_classifier.py "This course covers machine learning and neural networks."
```

Expected output format:

```text
Model: anthropic/claude-opus-4.6
Description: This course covers machine learning and neural networks.
Prediction:
  27  Computer science
```

### 6.3 Interactive mode

```bash
python llm_classifier.py
```

Then type a course description and press Enter.

Exit with:

```text
exit
quit
Ctrl-D
```

### 6.4 Change the LLM model

```bash
python llm_classifier.py "description..." --model google/gemini-2.5-flash-lite
```

### 6.5 Preview the prompt

```bash
python llm_classifier.py --dry-run
```

This prints the full system prompt without calling the API.

## 7. Metrics

The project reports three main metrics for the DDC-based models:

```text
DDC Micro F1
CNU Micro F1
CNU Subset Accuracy
```

### DDC Micro F1

Measures the quality of the intermediate DDC prediction step.

### CNU Micro F1

Measures the final CNU prediction quality after DDC-to-CNU mapping. This is
the main metric for comparing the classical-ML methods.

### CNU Subset Accuracy

Measures exact-match accuracy for the full predicted CNU label set. It is
stricter than Micro F1 because every predicted label must match the true set.

## 8. Suggested Experiment Workflow

Use this sequence when preparing results:

```bash
conda activate cnu_mapper

# Optional quick test
python ddc_classifier.py --compare --max-samples 5000

# Full evaluation table
python ddc_classifier.py --compare | tee eval_compare.txt

# Train final local models
python ddc_classifier.py --train --model-type logreg
python ddc_classifier.py --train --model-type mlp

# Test trained models on examples
python ddc_classifier.py --model-type logreg "Ce cours introduit l'apprentissage automatique et les reseaux de neurones."
python ddc_classifier.py --model-type mlp "Ce cours porte sur la chimie organique et la catalyse."
```

## 9. Example Inputs

Computer science:

```bash
python ddc_classifier.py --model-type logreg \
  "Ce cours introduit l'apprentissage automatique, les reseaux de neurones, la fouille de donnees et les algorithmes."
```

Law:

```bash
python ddc_classifier.py --model-type logreg \
  "Ce cours etudie le droit constitutionnel, le droit administratif, les institutions publiques et le contentieux."
```

Chemistry:

```bash
python ddc_classifier.py --model-type logreg \
  "Ce cours porte sur la chimie organique, la structure moleculaire, les reactions chimiques et la spectroscopie."
```

Linguistics:

```bash
python ddc_classifier.py --model-type logreg \
  "Ce cours introduit la phonetique, la syntaxe, la semantique, la morphologie et la variation linguistique."
```

## 10. GitHub Notes

The following files should not be committed to a normal GitHub repository:

```text
raw.csv
theses_labeled.csv
ddc_model_logreg.pkl
ddc_model_mlp.pkl
.env
__pycache__/
```

Reasons:

- `raw.csv` and `theses_labeled.csv` are large generated data files.
- `.pkl` files are generated trained models.
- `.env` contains a private API key.
- `__pycache__/` contains Python cache files.

If you want to publish data or trained models, use Git LFS, a release asset, or
an external storage service.

## 11. Troubleshooting

### `conda not found`

Install Miniconda or Anaconda, then run:

```bash
bash setup.sh
```

### `OPENROUTER_API_KEY not found`

Create `.env` in the project directory:

```bash
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' > .env
```

### `No module named openai`

Activate the project environment:

```bash
conda activate cnu_mapper
```

### Missing `theses_labeled.csv`

Run:

```bash
python label_theses.py
```

### Missing trained model file

Run the corresponding training command:

```bash
python ddc_classifier.py --train --model-type logreg
python ddc_classifier.py --train --model-type mlp
```
