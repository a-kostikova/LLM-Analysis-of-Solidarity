import json
import os
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


MODEL_NAME = "gpt-5-chat-latest"  # also used in the paper: gpt-4-1106-preview; gpt-3.5-turbo-0125
TARGET_CATEGORY = "Migrant"       # "Migrant" or "Frau"
TASK_MODE = "two_step"            # "one_step" or "two_step"
SHOT_MODE = "zeroshot"            # "zeroshot" or "fewshot"

PROMPTS_PATH = "ExperimentsScripts/prompts.json"
INPUT_FILE_PATH = "Data/Datasets/Migrant_1867-2025.json"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_response_column_names(model_name: str) -> tuple[str, str]:
    if model_name == "gpt-4-1106-preview":
        return "model_response_GPT4", "model_response_GPT4_subtype"
    if model_name == "gpt-3.5-turbo-0125":
        return "model_response_GPT3.5", "model_response_GPT3.5_subtype"
    if model_name in {"gpt-5-chat", "gpt-5-chat-latest"}:
        return "model_response_GPT5", "model_response_GPT5_subtype"
    raise ValueError(
        f"Unsupported MODEL_NAME '{model_name}'. "
        "Expected 'gpt-5-chat', 'gpt-5-chat-latest', "
        "'gpt-4-1106-preview', or 'gpt-3.5-turbo-0125'."
    )


def get_model_slug(model_name: str) -> str:
    if model_name == "gpt-4-1106-preview":
        return "gpt4"
    if model_name == "gpt-3.5-turbo-0125":
        return "gpt35"
    if model_name in {"gpt-5-chat", "gpt-5-chat-latest"}:
        return "gpt5"
    raise ValueError(f"Unsupported MODEL_NAME '{model_name}'.")


def validate_config(target_category: str, task_mode: str, shot_mode: str) -> None:
    if target_category not in {"Migrant", "Frau"}:
        raise ValueError("TARGET_CATEGORY must be 'Migrant' or 'Frau'.")
    if task_mode not in {"one_step", "two_step"}:
        raise ValueError("TASK_MODE must be 'one_step' or 'two_step'.")
    if shot_mode not in {"zeroshot", "fewshot"}:
        raise ValueError("SHOT_MODE must be 'zeroshot' or 'fewshot'.")


def derive_prompt_keys(
    target_category: str,
    task_mode: str,
    shot_mode: str,
) -> tuple[str, str | None, str | None]:
    if task_mode == "one_step":
        main_key = f"{target_category}_{shot_mode}"
        return main_key, None, None

    main_key = f"{target_category}_highlevel_{shot_mode}"
    solidarity_key = f"{target_category}_solidarity_{shot_mode}"
    antisolidarity_key = f"{target_category}_antisolidarity_{shot_mode}"
    return main_key, solidarity_key, antisolidarity_key


def needs_middle_tags(task_mode: str) -> bool:
    return task_mode == "two_step"


def derive_output_file_path(
    input_file_path: str,
    target_category: str,
    model_name: str,
    shot_mode: str,
    task_mode: str,
) -> str:
    input_path = Path(input_file_path)
    suffix = input_path.suffix.lower() or ".json"

    model_slug = get_model_slug(model_name)
    filename = f"{target_category}_{model_slug}_{shot_mode}{suffix}"

    return str(input_path.with_name(filename))


def load_prompts(prompts_path: str) -> dict:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def api_call(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def normalize_context_field(value) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def build_input_text(data_point: dict, add_middle_tags: bool) -> str:
    prev = data_point["context1"].strip()
    middle = data_point["input"].strip()
    next_ = data_point["context2"].strip()

    if add_middle_tags:
        parts = [prev, f"<MIDDLE>{middle}</MIDDLE>", next_]
    else:
        parts = [prev, middle, next_]

    return " ".join(part for part in parts if part)


def build_prompt(prompt_template: str, data_point: dict, add_middle_tags: bool) -> str:
    input_text = build_input_text(data_point, add_middle_tags)
    return f"{prompt_template}\n\n### Input Text:\n{input_text}"


def load_input_data(file_path: str) -> tuple[pd.DataFrame, str]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, delimiter=";")
        return df, "csv"

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df, "json"

    raise ValueError(f"Unsupported input format: {suffix}")


def save_output_data(df: pd.DataFrame, file_path: str, file_type: str) -> None:
    path = Path(file_path)

    if file_type == "csv":
        df.to_csv(path, index=False, sep=";")
        return

    if file_type == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        return

    raise ValueError(f"Unsupported output format: {file_type}")


def row_matches_category(row: pd.Series, file_type: str, target_category: str) -> bool:
    if file_type == "csv":
        row_category = str(row.get("Category", "")).strip().lower()
        return row_category == target_category.lower()

    if file_type == "json":
        row_category = str(row.get("category", "")).strip().lower()
        return row_category == target_category.lower()

    return False


def build_data_point(row: pd.Series, file_type: str) -> dict:
    if file_type == "csv":
        return {
            "context1": normalize_context_field(row.get("Previous", "")),
            "input": normalize_context_field(row.get("Middle", "")),
            "context2": normalize_context_field(row.get("Next", "")),
            "category": str(row.get("Category", "")),
        }

    if file_type == "json":
        return {
            "context1": normalize_context_field(row.get("prev_sents", "")),
            "input": normalize_context_field(row.get("sent", "")),
            "context2": normalize_context_field(row.get("next_sents", "")),
            "category": str(row.get("category", "")),
        }

    raise ValueError(f"Unsupported file type: {file_type}")


def extract_high_level_label(response_text: str) -> str | None:
    if not isinstance(response_text, str):
        return None

    match = re.search(
        r"LABEL\s*[:\-]?\s*(ANTI-SOLIDARITY|SOLIDARITY|MIXED|NONE)\b",
        response_text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().upper()

    upper_text = response_text.upper()
    for label in ["ANTI-SOLIDARITY", "SOLIDARITY", "MIXED", "NONE"]:
        if re.search(rf"\b{re.escape(label)}\b", upper_text):
            return label

    return None


def already_processed(row: pd.Series, high_col: str, sub_col: str | None, task_mode: str) -> bool:
    high_value = row.get(high_col)

    if not isinstance(high_value, str) or not high_value.strip():
        return False

    if task_mode == "one_step":
        return True

    high_label = extract_high_level_label(high_value)

    if high_label in {"NONE", "MIXED"}:
        return True

    if high_label in {"SOLIDARITY", "ANTI-SOLIDARITY"}:
        sub_value = row.get(sub_col) if sub_col else None
        return isinstance(sub_value, str) and bool(sub_value.strip())

    return False


def process_row(
    data_point: dict,
    prompts: dict,
    main_prompt_key: str,
    solidarity_prompt_key: str | None,
    antisolidarity_prompt_key: str | None,
    task_mode: str,
) -> tuple[str, str | None]:
    add_middle_tags = needs_middle_tags(task_mode)

    if main_prompt_key not in prompts:
        raise KeyError(f"Prompt key '{main_prompt_key}' not found in {PROMPTS_PATH}")

    main_prompt = prompts[main_prompt_key]
    main_prompt_full = build_prompt(main_prompt, data_point, add_middle_tags)
    high_response = api_call(main_prompt_full)

    if task_mode == "one_step":
        return high_response, None

    high_label = extract_high_level_label(high_response)
    subtype_response = None

    if high_label == "SOLIDARITY":
        if not solidarity_prompt_key or solidarity_prompt_key not in prompts:
            raise KeyError(f"Prompt key '{solidarity_prompt_key}' not found in {PROMPTS_PATH}")
        subtype_prompt = prompts[solidarity_prompt_key]
        subtype_response = api_call(build_prompt(subtype_prompt, data_point, add_middle_tags))

    elif high_label == "ANTI-SOLIDARITY":
        if not antisolidarity_prompt_key or antisolidarity_prompt_key not in prompts:
            raise KeyError(f"Prompt key '{antisolidarity_prompt_key}' not found in {PROMPTS_PATH}")
        subtype_prompt = prompts[antisolidarity_prompt_key]
        subtype_response = api_call(build_prompt(subtype_prompt, data_point, add_middle_tags))

    return high_response, subtype_response


def process_and_save_dataset(
    input_file_path: str,
    target_category: str,
    output_file_path: str,
    prompts_path: str,
    task_mode: str,
    shot_mode: str,
) -> None:
    validate_config(target_category, task_mode, shot_mode)

    df, file_type = load_input_data(input_file_path)
    prompts = load_prompts(prompts_path)

    main_prompt_key, solidarity_prompt_key, antisolidarity_prompt_key = derive_prompt_keys(
        target_category=target_category,
        task_mode=task_mode,
        shot_mode=shot_mode,
    )

    high_col, sub_col = get_response_column_names(MODEL_NAME)

    if high_col not in df.columns:
        df[high_col] = None

    if task_mode == "two_step" and sub_col not in df.columns:
        df[sub_col] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not row_matches_category(row, file_type, target_category):
            continue

        if already_processed(row, high_col, sub_col if task_mode == "two_step" else None, task_mode):
            continue

        try:
            data_point = build_data_point(row, file_type)
            high_response, subtype_response = process_row(
                data_point=data_point,
                prompts=prompts,
                main_prompt_key=main_prompt_key,
                solidarity_prompt_key=solidarity_prompt_key,
                antisolidarity_prompt_key=antisolidarity_prompt_key,
                task_mode=task_mode,
            )

            df.at[index, high_col] = high_response
            if task_mode == "two_step":
                df.at[index, sub_col] = subtype_response

        except Exception as e:
            print(f"Error processing row {index}: {e}")

        if index > 0 and index % 10 == 0:
            save_output_data(df, output_file_path, file_type)

    save_output_data(df, output_file_path, file_type)
    print(f"Final data saved to {output_file_path}.")


if __name__ == "__main__":
    output_file_path = derive_output_file_path(
        input_file_path=INPUT_FILE_PATH,
        target_category=TARGET_CATEGORY,
        model_name=MODEL_NAME,
        shot_mode=SHOT_MODE,
        task_mode=TASK_MODE,
    )

    process_and_save_dataset(
        input_file_path=INPUT_FILE_PATH,
        target_category=TARGET_CATEGORY,
        output_file_path=output_file_path,
        prompts_path=PROMPTS_PATH,
        task_mode=TASK_MODE,
        shot_mode=SHOT_MODE,
    )