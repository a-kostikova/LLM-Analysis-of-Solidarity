import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset


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
        logits = self.classifier(embeddings)
        return logits


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_id2label(task_dir):
    path = os.path.join(task_dir, "id2label.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing id2label.json in: {task_dir}")

    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def load_task_config(task_dir):
    config_path = os.path.join(task_dir, "config.json")
    if os.path.exists(config_path):
        return load_json(config_path)
    return {}


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


def load_model(model_checkpoint_path, config, num_labels, device):
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {model_checkpoint_path}")

    model_name = config.get(
        "model_type",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
    )
    dropout = config.get("dropout", 0.1)
    embedding_dim = config.get("embedding_dim", 512)

    model = SBERTClassifier(
        num_labels=num_labels,
        model_name=model_name,
        dropout=dropout,
        device=device,
        embedding_dim=embedding_dim,
    )

    state_dict = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_prediction_ids(model, dataloader):
    pred_ids = []

    with torch.no_grad():
        for batch in dataloader:
            sentences = batch["sentences"]
            logits = model(sentences)
            preds = torch.argmax(logits, dim=1)
            pred_ids.extend(preds.cpu().numpy().tolist())

    return pred_ids


def predict_with_task_model(df, model, config, id2label, batch_size):
    encoding = config.get("encoding", "target_only")
    add_category = config.get("add_category", False)

    texts = build_input_texts(
        df,
        encoding=encoding,
        add_category=add_category,
    )

    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pred_ids = get_prediction_ids(model, dataloader)
    pred_labels = [id2label[int(pred_id)] for pred_id in pred_ids]

    return pred_ids, pred_labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--major_task_dir", type=str, required=True)
    parser.add_argument("--s_task_dir", type=str, required=True)
    parser.add_argument("--as_task_dir", type=str, required=True)

    parser.add_argument("--major_checkpoint", type=str, required=True)
    parser.add_argument("--s_checkpoint", type=str, required=True)
    parser.add_argument("--as_checkpoint", type=str, required=True)

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--category_filter", type=str, default=None)
    parser.add_argument("--input_delimiter", type=str, default=";")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    major_id2label = load_id2label(args.major_task_dir)
    s_id2label = load_id2label(args.s_task_dir)
    as_id2label = load_id2label(args.as_task_dir)

    major_config = load_task_config(args.major_task_dir)
    s_config = load_task_config(args.s_task_dir)
    as_config = load_task_config(args.as_task_dir)

    major_model = load_model(
        args.major_checkpoint,
        config=major_config,
        num_labels=len(major_id2label),
        device=device,
    )
    s_model = load_model(
        args.s_checkpoint,
        config=s_config,
        num_labels=len(s_id2label),
        device=device,
    )
    as_model = load_model(
        args.as_checkpoint,
        config=as_config,
        num_labels=len(as_id2label),
        device=device,
    )

    df = pd.read_csv(args.input_csv, delimiter=args.input_delimiter)

    required_columns = ["Previous", "Middle", "Next", "Category"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if args.category_filter is not None:
        df = df[
            df["Category"].astype(str).str.contains(args.category_filter, case=False, na=False)
        ].copy()

    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df)} rows for inference.")

    _, major_pred_labels = predict_with_task_model(
        df,
        major_model,
        major_config,
        major_id2label,
        batch_size=args.batch_size,
    )

    df["High_Level_Category"] = major_pred_labels
    df["Predicted_Label"] = np.nan

    s_mask = df["High_Level_Category"] == "s"
    as_mask = df["High_Level_Category"] == "as"
    mixed_mask = df["High_Level_Category"] == "mixed"
    none_mask = df["High_Level_Category"] == "none"

    if s_mask.any():
        _, s_pred_labels = predict_with_task_model(
            df.loc[s_mask].copy(),
            s_model,
            s_config,
            s_id2label,
            batch_size=args.batch_size,
        )
        df.loc[s_mask, "Predicted_Label"] = s_pred_labels

    if as_mask.any():
        _, as_pred_labels = predict_with_task_model(
            df.loc[as_mask].copy(),
            as_model,
            as_config,
            as_id2label,
            batch_size=args.batch_size,
        )
        df.loc[as_mask, "Predicted_Label"] = as_pred_labels

    df.loc[mixed_mask, "Predicted_Label"] = "mixed.none"
    df.loc[none_mask, "Predicted_Label"] = "none.none"

    print("\nPrediction preview:")
    print(df[["High_Level_Category", "Predicted_Label"]].head(20))

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()