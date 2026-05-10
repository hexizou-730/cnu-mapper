"""
transformer_ddc_classifier.py
=============================

DDC-first classifier based on a pretrained Transformer model.

This script adds a PyTorch/Transformers route to the existing project without
replacing the TF-IDF + LogReg / MLP baselines in ddc_classifier.py.

Pipeline:

    abstract/resume -> XLM-RoBERTa -> DDC prediction -> DDC-to-CNU mapping

The model predicts DDC codes, not CNU codes directly. Final CNU labels and
metrics are produced through the same dewey_to_cnu.py mapping and the same
threshold + Top-K logic used by the sklearn baselines.

Usage:
    python transformer_ddc_classifier.py --eval --max-samples 5000
    python transformer_ddc_classifier.py --train --max-samples 5000
    python transformer_ddc_classifier.py "course description..."
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Allow PyTorch to fall back to CPU for rare operations that are not supported
# by Apple Silicon MPS yet.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer
    from transformers import logging as hf_logging
except ImportError as exc:
    sys.exit(
        "Missing Transformer dependencies.\n"
        "Install them with:\n"
        "  pip install -r requirements.txt\n"
        f"\nOriginal error: {exc}"
    )

hf_logging.set_verbosity_error()

from ddc_classifier import (
    RANDOM_SEED,
    TOP_K,
    VALIDATION_FRACTION,
    compute_metrics_from_ddc_binary,
    ddc_top_k_accuracy_from_proba,
    ddc_to_cnu_codes,
    load_data,
    load_kb,
    make_train_test_split,
    predict_binary_matrix,
    print_report_table,
    split_indices,
    tune_threshold,
)


HERE = Path(__file__).parent
DEFAULT_MODEL_NAME = "xlm-roberta-base"
MODEL_PATH = HERE / "ddc_model_xlm_roberta.pt"


def choose_device(requested: str) -> torch.device:
    """Return the best available PyTorch device for this machine."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Make training as reproducible as practical across CPU/GPU backends."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDDCDataset(Dataset):
    """Small dataset wrapper holding raw texts and multi-label DDC targets."""

    def __init__(self, texts: list[str], labels: np.ndarray | None = None):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx].astype("float32")
        return item


def make_collate_fn(tokenizer, max_length: int):
    """Create a DataLoader collator that tokenizes each text batch."""

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "labels" in batch[0]:
            encoded["labels"] = torch.tensor(
                np.stack([item["labels"] for item in batch]),
                dtype=torch.float32,
            )
        return encoded

    return collate


class TransformerDDCClassifier(nn.Module):
    """Pretrained encoder plus a multi-label DDC classification head."""

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_token))


@dataclass
class TrainConfig:
    """Training settings exposed through CLI flags."""

    model_name: str
    epochs: int
    batch_size: int
    lr: float
    max_length: int
    device: torch.device
    freeze_encoder: bool


def set_encoder_trainability(model: TransformerDDCClassifier, freeze_encoder: bool) -> None:
    """Optionally freeze the pretrained encoder and train only the classifier."""
    for param in model.encoder.parameters():
        param.requires_grad = not freeze_encoder


def train_model(
    model: TransformerDDCClassifier,
    tokenizer,
    texts: list[str],
    labels: np.ndarray,
    config: TrainConfig,
) -> None:
    """Train the Transformer DDC classifier for the configured number of epochs."""
    dataset = TextDDCDataset(texts, labels)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, config.max_length),
        num_workers=0,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr,
    )

    model.to(config.device)
    model.train()
    if config.freeze_encoder:
        model.encoder.eval()

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        total_items = 0

        for step, batch in enumerate(loader, start=1):
            labels_batch = batch.pop("labels").to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            batch_size = labels_batch.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size

            if step % 100 == 0:
                avg = total_loss / max(total_items, 1)
                print(f"   epoch {epoch} step {step:>5} avg loss {avg:.4f}", flush=True)

        avg_loss = total_loss / max(total_items, 1)
        print(
            f"   epoch {epoch} complete: loss {avg_loss:.4f}, "
            f"time {time.time() - t0:.1f}s"
        )


@torch.no_grad()
def predict_probabilities(
    model: TransformerDDCClassifier,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    """Return sigmoid probabilities for each DDC class."""
    dataset = TextDDCDataset(texts)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, max_length),
        num_workers=0,
    )
    model.to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        chunks.append(probs)
    return np.vstack(chunks)


def make_label_binarizer(labels: list[list[str]]) -> tuple[MultiLabelBinarizer, np.ndarray]:
    """Fit a MultiLabelBinarizer and return the binary label matrix."""
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    return mlb, y.astype("float32")


def restore_label_binarizer(classes: list[str]) -> MultiLabelBinarizer:
    """Rebuild a MultiLabelBinarizer from saved class names."""
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    return mlb


def save_checkpoint(
    path: Path,
    model: TransformerDDCClassifier,
    ddc_mlb: MultiLabelBinarizer,
    threshold: float,
    config: TrainConfig,
) -> None:
    """Save the trained model and the label metadata needed for prediction."""
    torch.save(
        {
            "model_name": config.model_name,
            "state_dict": model.cpu().state_dict(),
            "ddc_classes": list(ddc_mlb.classes_),
            "threshold": threshold,
            "top_k": TOP_K,
            "max_length": config.max_length,
            "freeze_encoder": config.freeze_encoder,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device):
    """Load a saved Transformer DDC checkpoint."""
    if not path.exists():
        sys.exit(
            f"Error: {path.name} not found.\n"
            "Train it first with: python transformer_ddc_classifier.py --train"
        )
    bundle = torch.load(path, map_location="cpu")
    model_name = bundle["model_name"]
    classes = list(bundle["ddc_classes"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TransformerDDCClassifier(model_name, len(classes))
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    ddc_mlb = restore_label_binarizer(classes)
    threshold = float(bundle["threshold"])
    max_length = int(bundle["max_length"])
    return tokenizer, model, ddc_mlb, threshold, max_length


def build_config(args) -> TrainConfig:
    """Translate parsed CLI flags into a typed training config."""
    device = choose_device(args.device)
    return TrainConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        device=device,
        freeze_encoder=args.freeze_encoder,
    )


def step_eval(args) -> None:
    """Train and evaluate XLM-R on the shared 80/20 split."""
    seed_everything(RANDOM_SEED)
    config = build_config(args)

    print(f"Loading data: theses_labeled.csv")
    texts, ddc_labels, cnu_labels = load_data(args.max_samples)
    if args.max_samples:
        print(f"   Max samples mode: {args.max_samples:,}")
    print(f"   Model: {config.model_name}")
    print(f"   Device: {config.device}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Max length: {config.max_length}")
    print(f"   Freeze encoder: {config.freeze_encoder}")

    _, fit_idx, val_idx, test_idx = make_train_test_split(len(texts))
    fit_texts = [texts[i] for i in fit_idx]
    fit_ddc = [ddc_labels[i] for i in fit_idx]
    val_texts = [texts[i] for i in val_idx]
    val_ddc = [ddc_labels[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_ddc = [ddc_labels[i] for i in test_idx]
    test_cnu = [cnu_labels[i] for i in test_idx]

    print(
        f"   Fit: {len(fit_texts):,}   "
        f"Validation: {len(val_texts):,}   "
        f"Test: {len(test_texts):,}"
    )

    ddc_mlb, y_fit = make_label_binarizer(fit_ddc)
    y_val = ddc_mlb.transform(val_ddc).astype("float32")
    y_test = ddc_mlb.transform(test_ddc)

    cnu_mlb = MultiLabelBinarizer()
    cnu_mlb.fit(cnu_labels)
    y_cnu_test = cnu_mlb.transform(test_cnu)

    print(f"   DDC classes: {len(ddc_mlb.classes_)}")
    print("Loading pretrained Transformer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = TransformerDDCClassifier(config.model_name, len(ddc_mlb.classes_))
    set_encoder_trainability(model, config.freeze_encoder)

    print("Training DDC Transformer...")
    t0 = time.time()
    train_model(model, tokenizer, fit_texts, y_fit, config)
    print(f"   Training time: {time.time() - t0:.1f}s")

    print("Tuning DDC threshold...")
    val_proba = predict_probabilities(
        model, tokenizer, val_texts, config.batch_size,
        config.max_length, config.device
    )
    threshold, val_micro_f1 = tune_threshold(val_proba, y_val)
    print(f"   Best threshold: {threshold:.3f}")
    print(f"   Validation DDC micro-F1: {val_micro_f1:.4f}")

    print("\nEvaluating...")
    test_proba = predict_probabilities(
        model, tokenizer, test_texts, config.batch_size,
        config.max_length, config.device
    )
    y_pred = predict_binary_matrix(test_proba, threshold)
    row = compute_metrics_from_ddc_binary(
        "XLM-R", y_pred, y_test, y_cnu_test, ddc_mlb, cnu_mlb, threshold,
        ddc_top1_accuracy=ddc_top_k_accuracy_from_proba(test_proba, y_test, 1),
        ddc_top3_accuracy=ddc_top_k_accuracy_from_proba(test_proba, y_test, 3),
    )
    print_report_table([row])


def step_train(args) -> None:
    """Train and save a reusable XLM-R DDC model checkpoint."""
    seed_everything(RANDOM_SEED)
    config = build_config(args)

    print(f"Loading data: theses_labeled.csv")
    texts, ddc_labels, _ = load_data(args.max_samples)
    if args.max_samples:
        print(f"   Max samples mode: {args.max_samples:,}")
    print(f"   Model: {config.model_name}")
    print(f"   Device: {config.device}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Max length: {config.max_length}")
    print(f"   Freeze encoder: {config.freeze_encoder}")

    fit_idx, val_idx = split_indices(
        len(texts), 1.0 - VALIDATION_FRACTION, RANDOM_SEED
    )
    fit_texts = [texts[i] for i in fit_idx]
    fit_ddc = [ddc_labels[i] for i in fit_idx]
    val_texts = [texts[i] for i in val_idx]
    val_ddc = [ddc_labels[i] for i in val_idx]
    print(f"   Fit: {len(fit_texts):,}   Validation: {len(val_texts):,}")

    ddc_mlb, y_fit = make_label_binarizer(fit_ddc)
    y_val = ddc_mlb.transform(val_ddc).astype("float32")
    print(f"   DDC classes: {len(ddc_mlb.classes_)}")

    print("Loading pretrained Transformer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = TransformerDDCClassifier(config.model_name, len(ddc_mlb.classes_))
    set_encoder_trainability(model, config.freeze_encoder)

    print("Training DDC Transformer...")
    t0 = time.time()
    train_model(model, tokenizer, fit_texts, y_fit, config)
    print(f"   Training time: {time.time() - t0:.1f}s")

    print("Tuning DDC threshold...")
    val_proba = predict_probabilities(
        model, tokenizer, val_texts, config.batch_size,
        config.max_length, config.device
    )
    threshold, val_micro_f1 = tune_threshold(val_proba, y_val)
    print(f"   Best threshold: {threshold:.3f}")
    print(f"   Validation DDC micro-F1: {val_micro_f1:.4f}")

    print(f"Saving model: {MODEL_PATH.name}")
    save_checkpoint(MODEL_PATH, model, ddc_mlb, threshold, config)
    print(f"   Size: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print("Training complete.")


def predict_one(args) -> None:
    """Load a saved checkpoint and predict one free-text description."""
    device = choose_device(args.device)
    tokenizer, model, ddc_mlb, threshold, max_length = load_checkpoint(
        MODEL_PATH, device
    )
    proba = predict_probabilities(
        model, tokenizer, [args.text], args.batch_size, max_length, device
    )
    y_pred = predict_binary_matrix(proba, threshold)
    ranked = np.argsort(-proba[0])
    selected = set(np.where(y_pred[0])[0])
    ddc_codes = [ddc_mlb.classes_[j] for j in ranked if j in selected]
    cnu_codes = ddc_to_cnu_codes(ddc_codes)
    by_code = load_kb()

    print("Model: DDC XLM-R")
    print(f"Decision threshold: {threshold:.3f}")
    print(f"Description: {args.text}")
    print(f"Predicted DDC: {', '.join(ddc_codes)}")
    print("Mapped CNU:")
    for cnu in cnu_codes:
        print(f"  {cnu}  {by_code.get(cnu, '?')}")


def main() -> None:
    """Parse CLI arguments and run training, evaluation, or prediction."""
    parser = argparse.ArgumentParser(
        description="Predict DDC with XLM-RoBERTa, then map DDC to CNU."
    )
    parser.add_argument("text", nargs="?", help="Course description to classify.")
    parser.add_argument("--train", action="store_true",
                        help="Train and save the XLM-R DDC model.")
    parser.add_argument("--eval", action="store_true",
                        help="Train/evaluate XLM-R on the shared 80/20 split.")
    parser.add_argument("--max-samples", type=int,
                        help="Optional small-sample run for quick experiments.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="Hugging Face model name.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size. Try 4 if MPS memory is tight.")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum tokenizer length in subword tokens.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="AdamW learning rate.")
    parser.add_argument("--device", default="auto",
                        help="auto, mps, cuda, or cpu.")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Train only the classification head.")
    args = parser.parse_args()

    if args.train:
        step_train(args)
        return
    if args.eval:
        step_eval(args)
        return
    if args.text:
        predict_one(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
