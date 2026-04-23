import json
import math
import re
from pathlib import Path

import pandas as pd


INPUT_FILE = "Migrant18kGPT4_predicted.json"  # .json or .csv
MODEL_NAME = "GPT4"  # e.g. "GPT5", "GPT4", "GPT3.5", "Llama3", "Qwen25"
TASK_MODE = "one_step"  # "one_step" or "two_step"


LABEL2SHORT = {
    "NONE": "none.none",
    "MIXED": "mixed.none",
    "GROUP-BASED SOLIDARITY": "s.group-based",
    "EXCHANGE-BASED SOLIDARITY": "s.exchange-based",
    "EMPATHIC SOLIDARITY": "s.empathic",
    "COMPASSIONATE SOLIDARITY": "s.compassionate",
    "GROUP-BASED ANTI-SOLIDARITY": "as.group-based",
    "EXCHANGE-BASED ANTI-SOLIDARITY": "as.exchange-based",
    "EMPATHIC ANTI-SOLIDARITY": "as.empathic",
    "COMPASSIONATE ANTI-SOLIDARITY": "as.compassionate",
    "SOLIDARITY": "s.none",
    "ANTI-SOLIDARITY": "as.none",
}


def validate_config(model_name: str, task_mode: str) -> None:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("MODEL_NAME must be a non-empty string.")
    if task_mode not in {"one_step", "two_step"}:
        raise ValueError("TASK_MODE must be 'one_step' or 'two_step'.")


def get_model_columns(model_name: str) -> tuple[str, str, str]:
    model_name = model_name.strip()
    response_col = f"model_response_{model_name}"
    label_col = f"extracted_label_{model_name}"
    subtype_response_col = f"model_response_{model_name}_subtype"
    return response_col, label_col, subtype_response_col


def normalize_text(text) -> str:
    if not isinstance(text, str):
        return ""

    replacements = {
        "\u2011": "-",
        "\u2010": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00A0": " ",
        "\u2022": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\*\*", " ", text)
    text = re.sub(r"__+", " ", text)
    return text


def normalize_label_candidate(text: str) -> str:
    text = normalize_text(text)
    text = text.strip(" \t\n\r .;:[](){}*_")
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return str(value).strip() == ""


def extract_raw_label(response: str) -> str | None:
    if not isinstance(response, str):
        return None

    lines = response.splitlines()

    for i, line in enumerate(lines):
        same_line_match = re.search(r"LABEL\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if same_line_match:
            candidate = normalize_label_candidate(same_line_match.group(1))
            return candidate if candidate else None

        header_only_match = re.search(r"LABEL\s*:\s*$", line, flags=re.IGNORECASE)
        if header_only_match:
            for next_line in lines[i + 1:]:
                if str(next_line).strip():
                    candidate = normalize_label_candidate(next_line)
                    return candidate if candidate else None
            return None

    return None


def derive_final_label(main_label: str | None, subtype_label: str | None = None) -> str:
    if main_label is None:
        return "label not found"

    main_label = main_label.upper().strip()

    if main_label in {"NONE", "MIXED"}:
        return LABEL2SHORT.get(main_label, "label not found")

    if main_label == "SOLIDARITY":
        if subtype_label:
            subtype_label = subtype_label.upper().strip()
            if subtype_label in {
                "GROUP-BASED SOLIDARITY",
                "EXCHANGE-BASED SOLIDARITY",
                "EMPATHIC SOLIDARITY",
                "COMPASSIONATE SOLIDARITY",
            }:
                return LABEL2SHORT[subtype_label]
        return LABEL2SHORT["SOLIDARITY"]

    if main_label == "ANTI-SOLIDARITY":
        if subtype_label:
            subtype_label = subtype_label.upper().strip()
            if subtype_label in {
                "GROUP-BASED ANTI-SOLIDARITY",
                "EXCHANGE-BASED ANTI-SOLIDARITY",
                "EMPATHIC ANTI-SOLIDARITY",
                "COMPASSIONATE ANTI-SOLIDARITY",
            }:
                return LABEL2SHORT[subtype_label]
        return LABEL2SHORT["ANTI-SOLIDARITY"]

    if main_label in LABEL2SHORT:
        return LABEL2SHORT[main_label]

    return "label not found"


def load_data(file_path: str) -> tuple[pd.DataFrame, str]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, delimiter=";"), "csv"

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f)), "json"

    raise ValueError(f"Unsupported input format: {suffix}")


def save_data(df: pd.DataFrame, file_path: str, file_type: str) -> None:
    path = Path(file_path)

    if file_type == "csv":
        df.to_csv(path, index=False, sep=";")
        return

    if file_type == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        return

    raise ValueError(f"Unsupported output format: {file_type}")


def extract_labels_from_file(input_file: str, model_name: str, task_mode: str) -> None:
    validate_config(model_name, task_mode)

    df, file_type = load_data(input_file)

    response_col, label_col, subtype_response_col = get_model_columns(model_name)

    if response_col not in df.columns:
        raise KeyError(
            f"Response column '{response_col}' not found in {input_file}"
        )

    use_subtype = task_mode == "two_step"

    if use_subtype and subtype_response_col not in df.columns:
        raise KeyError(
            f"Subtype response column '{subtype_response_col}' not found in {input_file}"
        )

    if label_col not in df.columns:
        df[label_col] = None

    processed = 0
    not_found = 0

    for idx, row in df.iterrows():
        response_text = row.get(response_col)

        if is_missing(response_text):
            continue

        main_label = extract_raw_label(response_text)

        if use_subtype:
            subtype_text = row.get(subtype_response_col)
            subtype_label = None if is_missing(subtype_text) else extract_raw_label(subtype_text)
            final_label = derive_final_label(main_label, subtype_label)
        else:
            final_label = derive_final_label(main_label)

        df.at[idx, label_col] = final_label
        processed += 1

        if final_label == "label not found":
            not_found += 1

    save_data(df, input_file, file_type)

    pct = (not_found / processed * 100) if processed else 0.0
    print(f"Model: {model_name}")
    print(f"Task mode: {task_mode}")
    print(f"Processed {processed} responses.")
    print(f"label not found: {not_found} ({pct:.2f}%)")
    print(f"Saved updated file: {input_file}")


if __name__ == "__main__":
    extract_labels_from_file(INPUT_FILE, MODEL_NAME, TASK_MODE)