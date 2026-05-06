# CNU Mapper

CNU Mapper is a course-description classifier for French CNU sections
(`Conseil National des Universites`). It compares three approaches:

1. A DDC-based One-vs-Rest Logistic Regression classifier.
2. A DDC-based Multi-Layer Perceptron classifier.
3. A prompt-based LLM classifier via OpenRouter.

The main classical-ML pipeline predicts Dewey Decimal Classification
(DDC) codes first, then maps the predicted DDC codes to CNU sections with a
rule-based mapping table.

```text
course description -> DDC prediction -> DDC-to-CNU mapping -> CNU sections
```

## Project Structure

```text
cnu_mapper/
|-- ddc_classifier.py                  # DDC-based LogReg and MLP classifiers
|-- llm_classifier.py                  # Prompt-based LLM classifier
|-- dewey_to_cnu.py                    # Rule-based DDC-to-CNU mapping table
|-- label_theses.py                    # Build labeled training data from raw.csv
|-- fetch_theses.py                    # Download / inspect / filter theses.fr data
|-- cnu_knowledge_base_v2.json         # Enriched runtime CNU knowledge base
|-- cnu_knowledge_base_official.json   # Official CNU reference data
|-- DATA_PROVENANCE.md                 # Data source notes
|-- USAGE_GUIDE.md                     # Detailed usage guide
`-- old_baseline/                      # Archived earlier baseline and diagnostics
```

Large local artifacts are intentionally ignored by Git:

```text
raw.csv
theses_labeled.csv
ddc_model_logreg.pkl
ddc_model_mlp.pkl
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

### 3. LLM Classifier

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

Evaluate the two DDC-based classical-ML methods:

```bash
python ddc_classifier.py --compare
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
  27  Computer science
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
Model    DDC Micro F1   CNU Micro F1   CNU Subset Accuracy
------   ------------   ------------   -------------------
LogReg   ...
MLP      ...
```

These metrics are used because the task is multi-label:

- `DDC Micro F1`: quality of the intermediate DDC prediction step.
- `CNU Micro F1`: main final-task metric after DDC-to-CNU mapping.
- `CNU Subset Accuracy`: strict exact-match score for the full CNU label set.

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
.env
```

## Documentation

See `USAGE_GUIDE.md` for detailed commands and workflow explanations.
See `DATA_PROVENANCE.md` for data-source notes.
