from argparse import ArgumentParser
from collections import Counter
import json
import math
import os
import pprint
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup


os.environ["WANDB_DISABLED"] = "true"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def oversample(df, on="label"):
    """
    Randomly sample minority classes until all classes match the majority size.
    """
    max_size = df[on].value_counts().max()
    chunks = [df]

    print(f"Oversampling on column: {on}")
    for _, group in df.groupby(on):
        if len(group) < max_size:
            chunks.append(
                group.sample(max_size - len(group), replace=True, random_state=42)
            )

    df_new = pd.concat(chunks)
    df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_new


def prepare_split(df, task):
    df = df.copy()

    if task == "major":
        target_col = "major"

    elif task == "s":
        df = df[df["major"] == "s"]
        df = df[df["label"].astype(str).str.startswith("s.")]
        df = df[~df["label"].astype(str).str.contains("s.none", case=False, na=False)]
        target_col = "label"

    elif task == "as":
        df = df[df["major"] == "as"]
        df = df[df["label"].astype(str).str.startswith("as.")]
        df = df[~df["label"].astype(str).str.contains("as.none", case=False, na=False)]
        target_col = "label"

    else:
        raise ValueError(f"Unknown task: {task}")

    df = df[df[target_col].notna()].reset_index(drop=True)
    return df, target_col


def build_input_texts(df, encoding="target_only", add_category=False):
    if encoding == "target_only" and not add_category:
        texts = df["Middle"].astype(str).tolist()

    elif encoding == "target_only" and add_category:
        texts = (
            df["Category"].astype(str) + " [SEP] " + df["Middle"].astype(str)
        ).tolist()

    elif encoding == "target+context" and not add_category:
        texts = df.apply(
            lambda x: " [SEP] ".join(
                [str(x["Previous"]), str(x["Middle"]), str(x["Next"])]
            ),
            axis=1,
        ).tolist()

    elif encoding == "target+context" and add_category:
        texts = df.apply(
            lambda x: str(x["Category"]) + " [SEP] " + " [SEP] ".join(
                [str(x["Previous"]), str(x["Middle"]), str(x["Next"])]
            ),
            axis=1,
        ).tolist()

    else:
        raise ValueError(f"Unsupported encoding setting: {encoding}")

    return texts


class TextDataset(Dataset):
    def __init__(self, sentences, labels=None):
        self.sentences = sentences
        self.labels = labels

    def __getitem__(self, idx):
        item = {"sentences": self.sentences[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.sentences)


class SBERTClassifier(nn.Module):
    def __init__(
        self,
        num_labels=4,
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        dropout=0.1,
        device="cuda",
        embedding_dim=512,
    ):
        super().__init__()
        self.device_name = device
        self.encoder = SentenceTransformer(model_name, device=device)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(self, sentences):
        embeddings = self.encoder.encode(
            sentences,
            convert_to_tensor=True,
            device=self.device_name,
            show_progress_bar=False,
        )

        embeddings = embeddings.clone().detach()
        logits = self.classifier(embeddings)
        return logits


def compute_metrics_from_preds(true_ids, pred_ids, id2label):
    true_labels = [id2label[int(label_id)] for label_id in true_ids]
    pred_labels = [id2label[int(label_id)] for label_id in pred_ids]

    print()
    print(classification_report(y_true=true_labels, y_pred=pred_labels, digits=6))

    report = classification_report(
        y_true=true_labels,
        y_pred=pred_labels,
        digits=6,
        output_dict=True,
        zero_division=0,
    )
    return report["macro avg"]


def evaluate(model, dataloader, criterion, id2label):
    model.eval()

    total_loss = 0.0
    total_items = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            sentences = batch["sentences"]
            labels = batch["labels"].to(model.device_name)

            logits = model(sentences)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / total_items if total_items > 0 else 0.0
    metrics = compute_metrics_from_preds(all_labels, all_preds, id2label)
    metrics["loss"] = avg_loss

    return metrics


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler):
    model.train()

    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        optimizer.zero_grad()

        sentences = batch["sentences"]
        labels = batch["labels"].to(model.device_name)

        logits = model(sentences)
        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    avg_loss = total_loss / total_items if total_items > 0 else 0.0
    return avg_loss


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--task", type=str, choices=["major", "s", "as"], required=True)
    parser.add_argument("--id", type=int, default=0, help="A unique id assigned to this training experiment")
    parser.add_argument("--data_dir", type=str, default="data/training_splits")
    parser.add_argument(
        "--model_type",
        type=str,
        default="sentence-transformers/distiluse-base-multilingual-cased-v2",
    )
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["target_only", "target+context"],
        default="target_only",
    )
    parser.add_argument("--add_category", action="store_true")
    parser.add_argument("--oversample", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20, help="Maximal number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="The initial learning rate for AdamW")
    parser.add_argument("--seed", type=int, default=8888, help="Random seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--not_save_checkpoint", action="store_true")
    parser.add_argument("--best_metric", type=str, default="eval_f1-score")

    global_args = parser.parse_args()
    pprint.pprint(vars(global_args), indent=4)

    if global_args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    seed_everything(global_args.seed)

    out_dir = os.path.join(global_args.save_dir, global_args.task, str(global_args.id))
    os.makedirs(out_dir, exist_ok=True)

    save_json(vars(global_args), os.path.join(out_dir, "config.json"))

    train_df = pd.read_csv(os.path.join(global_args.data_dir, "train.csv"), delimiter=";")
    dev_df = pd.read_csv(os.path.join(global_args.data_dir, "dev.csv"), delimiter=";")
    test_df = pd.read_csv(os.path.join(global_args.data_dir, "test.csv"), delimiter=";")

    train_df, target_col = prepare_split(train_df, global_args.task)
    dev_df, _ = prepare_split(dev_df, global_args.task)
    test_df, _ = prepare_split(test_df, global_args.task)

    print()
    print(f"Task: {global_args.task}")
    print(f"Target column: {target_col}")
    print("Train distribution:", Counter(train_df[target_col]))
    print("Dev distribution:", Counter(dev_df[target_col]))
    print("Test distribution:", Counter(test_df[target_col]))

    if global_args.oversample:
        train_df = oversample(train_df, on=target_col)
        print("Train distribution after oversampling:", Counter(train_df[target_col]))

    label2id = {
        label: i for i, label in enumerate(sorted(train_df[target_col].dropna().unique()))
    }
    id2label = {v: k for k, v in label2id.items()}

    save_json(label2id, os.path.join(out_dir, "label2id.json"))
    save_json(id2label, os.path.join(out_dir, "id2label.json"))

    train_texts = build_input_texts(
        train_df,
        encoding=global_args.encoding,
        add_category=global_args.add_category,
    )
    dev_texts = build_input_texts(
        dev_df,
        encoding=global_args.encoding,
        add_category=global_args.add_category,
    )
    test_texts = build_input_texts(
        test_df,
        encoding=global_args.encoding,
        add_category=global_args.add_category,
    )

    train_labels = torch.tensor(
        [label2id[label] for label in train_df[target_col].tolist()],
        dtype=torch.long,
    )
    dev_labels = torch.tensor(
        [label2id[label] for label in dev_df[target_col].tolist()],
        dtype=torch.long,
    )
    test_labels = torch.tensor(
        [label2id[label] for label in test_df[target_col].tolist()],
        dtype=torch.long,
    )

    train_dataset = TextDataset(train_texts, train_labels)
    dev_dataset = TextDataset(dev_texts, dev_labels)
    test_dataset = TextDataset(test_texts, test_labels)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=global_args.batch_size,
        shuffle=True,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=global_args.batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=global_args.batch_size,
        shuffle=False,
    )

    model = SBERTClassifier(
        num_labels=len(label2id),
        model_name=global_args.model_type,
        dropout=global_args.dropout,
        device=global_args.device,
        embedding_dim=global_args.embedding_dim,
    )
    model.to(global_args.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=global_args.learning_rate,
        eps=1e-8,
        weight_decay=global_args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_dataset) / global_args.batch_size)
    total_steps = steps_per_epoch * global_args.epochs
    actual_warmup_steps = (
        math.ceil(total_steps * global_args.warmup_ratio)
        if global_args.warmup_steps == -1
        else global_args.warmup_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=actual_warmup_steps,
        num_training_steps=total_steps,
    )

    best_metric_value = -float("inf")
    best_model_path = os.path.join(out_dir, "best_model.pt")

    for epoch in range(global_args.epochs):
        print()
        print(f"Epoch {epoch + 1}/{global_args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"Train loss: {train_loss:.6f}")

        print("\nDev evaluation:")
        dev_metrics = evaluate(
            model=model,
            dataloader=dev_dataloader,
            criterion=criterion,
            id2label=id2label,
        )

        metric_name = global_args.best_metric.replace("eval_", "")
        if metric_name not in dev_metrics:
            raise ValueError(
                f"Metric '{global_args.best_metric}' not found in dev metrics. "
                f"Available keys: {list(dev_metrics.keys())}"
            )

        current_metric_value = dev_metrics[metric_name]
        print(f"Tracked dev metric ({global_args.best_metric}): {current_metric_value:.6f}")

        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to: {best_model_path}")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=global_args.device))
        model.to(global_args.device)
        print(f"\nLoaded best model from: {best_model_path}")

    print("\nFinal dev evaluation:")
    evaluate(
        model=model,
        dataloader=dev_dataloader,
        criterion=criterion,
        id2label=id2label,
    )

    print("\nTest evaluation:")
    evaluate(
        model=model,
        dataloader=test_dataloader,
        criterion=criterion,
        id2label=id2label,
    )

    if global_args.not_save_checkpoint:
        shutil.rmtree(out_dir, ignore_errors=True)
        print("Deleted checkpoint directory.")