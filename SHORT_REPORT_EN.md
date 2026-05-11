# Short Report: CNU Mapper

## 1. Project Objective

This project aims to classify academic text descriptions into French CNU
sections (`Conseil National des Universites`). The input is an academic
abstract or course description, and the expected output is one or more CNU
section codes.

The main implemented pipeline follows a DDC-first design:

```text
text description -> DDC prediction -> DDC-to-CNU mapping -> CNU sections
```

This design was chosen because the source thesis dataset contains Dewey Decimal
Classification (DDC) metadata. Therefore, the supervised learning target is
DDC. CNU sections are not directly annotated in the source dataset; they are
derived from DDC codes using a rule-based DDC-to-CNU mapping table.

## 2. Dataset

The experiments use a generated local dataset, `theses_labeled.csv`, built from
the official French thesis dataset. The generated CSV is not committed to
GitHub because it is a large local artifact.

The preprocessing pipeline is implemented in `label_theses.py`. It keeps French
abstracts between 100 and 4000 characters, removes near-duplicate abstracts
using the first 200 characters, extracts DDC codes from the source metadata,
and maps DDC codes to CNU sections using `dewey_to_cnu.py`.

The resulting dataset is weakly labeled for CNU: DDC labels come from the
source metadata, while CNU labels are derived from the DDC-to-CNU mapping
rather than manually annotated by domain experts.

## 3. Dataset Statistics

| Item | Value |
|---|---:|
| Samples | 331,608 |
| Unique DDC codes | 92 |
| Official CNU sections | 80 |
| CNU sections appearing in the dataset | 45 |
| Average DDC labels per sample | 1.000 |
| Average CNU labels per sample | 1.240 |
| Median abstract length | 1,655 characters |
| Mean abstract length | 1,706 characters |

The DDC label is single-label in the generated dataset. The final CNU output is
multi-label because one DDC code can map to one or two CNU sections.

The dataset is highly imbalanced. The most common DDC codes are concentrated in
engineering, physics, biology, chemistry, law, medicine, computer science, and
psychology.

Top DDC codes:

| DDC code | Samples | Percentage |
|---:|---:|---:|
| 620 | 40,541 | 12.23% |
| 530 | 37,313 | 11.25% |
| 570 | 33,202 | 10.01% |
| 540 | 23,215 | 7.00% |
| 340 | 19,229 | 5.80% |
| 610 | 18,378 | 5.54% |
| 004 | 17,582 | 5.30% |
| 150 | 15,239 | 4.60% |

Top mapped CNU sections:

| CNU code | Section | Samples | Percentage |
|---:|---|---:|---:|
| 62 | Energy engineering and process engineering | 45,882 | 13.84% |
| 28 | Dense media and materials | 38,567 | 11.63% |
| 64 | Biochemistry and molecular biology | 33,202 | 10.01% |
| 65 | Cell biology | 33,202 | 10.01% |
| 32 | Organic, inorganic and industrial chemistry | 23,215 | 7.00% |
| 02 | Public law | 19,327 | 5.83% |
| 01 | Private law and criminal sciences | 19,229 | 5.80% |
| 27 | Computer science | 18,696 | 5.64% |

The generated dataset covers 45 of the 80 official CNU sections. The remaining
35 official CNU sections have no samples in the current weakly labeled data.
This is an important limitation because the supervised models can only learn
labels represented in the training data.

More detailed statistics are available in `DATASET_STATISTICS.md`.

## 4. Implemented Approaches

### 4.1 Random Baseline

The Random baseline predicts one uniformly random DDC code from the training
label space for each test example. It is included as a lower bound to show what
performance looks like without learning a relationship between text and labels.

### 4.2 Majority Baseline

The Majority baseline always predicts the most frequent DDC code observed in
the training split. This baseline is useful because the dataset is imbalanced:
a trivial majority predictor can already obtain a non-zero score.

### 4.3 DDC + Logistic Regression

The first trainable model uses TF-IDF features and One-vs-Rest Logistic
Regression. It trains one binary classifier per DDC code. The predicted DDC
codes are then mapped to CNU sections.

Important settings:

- TF-IDF word unigrams and bigrams
- Maximum vocabulary size: 100,000
- `min_df=3`
- Logistic Regression with `C=2.0`
- Class-balanced binary classifiers
- Global threshold calibrated on a validation split

### 4.4 DDC + Multi-Layer Perceptron

The second trainable model uses TF-IDF features and an `MLPClassifier` from
scikit-learn. Unlike the Logistic Regression approach, the MLP predicts the
complete DDC indicator vector with one neural network.

Important settings:

- TF-IDF word unigrams
- Maximum vocabulary size: 20,000
- One hidden layer with 64 units
- Adam optimizer
- Global threshold calibrated on a validation split

### 4.5 DDC + XLM-RoBERTa Transformer

The project also includes a PyTorch/Transformers route based on
`xlm-roberta-base`. This model predicts DDC codes first and then uses the same
DDC-to-CNU mapping as the classical machine-learning models.

The model uses the XLM-R encoder, the representation of the first token, a
dropout layer, and a linear multi-label DDC classification head. It is trained
with `BCEWithLogitsLoss`, then sigmoid probabilities are converted into DDC
predictions using the same threshold and Top-K fallback logic as the other DDC
models.

The completed full-data evaluation used one epoch, `batch-size=16`,
`max-length=256`, and Apple Silicon MPS acceleration.

### 4.6 LLM Classifier

The project also includes an LLM-based classifier using OpenRouter. This method
does not train a local model. It builds a prompt from the official CNU section
list and asks the LLM to return one to three CNU codes in JSON format.

The LLM method is implemented for qualitative comparison and manual testing. It
is not included in the full numerical evaluation table because evaluating it on
the full test set would require many external API calls.

## 5. Evaluation Protocol

The numerical evaluation uses an 80/20 train-test split. For trainable models,
the training portion is further split into fit and validation subsets. The
validation subset is used to calibrate one global DDC probability threshold.

For the full evaluation:

- Baseline train split: 265,286 samples
- Fit split for trainable models: 212,228 samples
- Validation split for threshold tuning: 53,058 samples
- Test split: 66,322 samples
- XLM-R evaluation setting: 1 epoch, batch size 16, max length 256
- XLM-R training time on Apple Silicon MPS: 39,541.8 seconds, about 10.98 hours
- XLM-R final training loss: 0.0298
- XLM-R calibrated threshold: 0.325
- XLM-R validation DDC micro-F1: 0.6186

The reported metrics are:

- **DDC Top-1 Accuracy**: whether the highest-scoring DDC candidate is correct.
- **DDC Top-3 Accuracy**: whether the true DDC appears among the top three DDC candidates.
- **Mapped CNU Micro F1**: proxy downstream quality after DDC-to-CNU mapping.

DDC metrics are the primary supervised evaluation metrics because DDC codes are
present in the source metadata. Mapped CNU Micro F1 is useful for understanding
the downstream application, but it should not be interpreted as direct
evaluation against manually annotated CNU ground truth.

## 6. Results

| Model | DDC Top-1 Acc | DDC Top-3 Acc | Mapped CNU Micro F1 |
|---|---:|---:|---:|
| Random | 1.12% | 1.12% | 0.0257 |
| Majority | 12.29% | 12.29% | 0.1247 |
| LogReg | 62.72% | 89.13% | 0.6650 |
| MLP | 52.21% | 77.27% | 0.5663 |
| XLM-R | 60.95% | 87.66% | 0.6542 |

The Logistic Regression model performs best among the evaluated approaches. It
substantially improves over both simple baselines, which indicates that the
model learns meaningful disciplinary signals from the abstracts.

The XLM-RoBERTa model is close to Logistic Regression and clearly better than
the MLP on all three reported metrics. However, it does not surpass Logistic
Regression in the current setup. One likely reason is that the Transformer
model only reads the first 256 subword tokens, while TF-IDF uses information
from the full abstract. Sparse TF-IDF features are also very effective for this
metadata-derived DDC classification task.

The MLP also improves over the simple baselines, but it performs worse than
both Logistic Regression and XLM-RoBERTa. The result suggests that adding model
complexity alone is not enough when the input representation remains sparse
TF-IDF.

## 7. Discussion and Limitations

The results should be interpreted in the context of the dataset construction.
DDC labels are available from metadata and can be evaluated directly. CNU labels
are mapped from DDC rather than manually annotated, so CNU scores reflect both
DDC prediction quality and the limitations of the DDC-to-CNU mapping.

The dataset is also imbalanced. Some mapped CNU sections have tens of thousands
of examples, while others are rare or absent. This affects both model training
and the interpretation of aggregate metrics.

For the Transformer model, the largest modeling limitation is input truncation:
`max-length=256` means that later parts of long abstracts are not visible to the
model. This may partly explain why XLM-RoBERTa is close to, but still below,
the Logistic Regression baseline.

Finally, only 45 of the 80 official CNU sections are represented in the current
training data. The current supervised models cannot learn sections that never
appear in the weak labels.

## 8. Next Steps

Possible next steps include:

- Refine the DDC-to-CNU mapping for ambiguous DDC codes.
- Add more detailed per-class evaluation, especially for rare CNU sections.
- Evaluate the LLM classifier on a smaller manually selected test set.
- Continue transformer-based experiments with longer text handling.
- Expand the final report with examples of correct and incorrect predictions.
