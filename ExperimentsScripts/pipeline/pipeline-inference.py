import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

MODEL_NAME = "Qwen25"  # "Llama3" or "Qwen25"
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"  # "meta-llama/Llama-3.3-70B-Instruct" or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN_ENV = "HF_TOKEN"

TARGET_CATEGORY = "Migrant"  # "Migrant" or "Frau"
SHOT_MODE = "fewshot"        # "zeroshot" or "fewshot"

INPUT_FILE = "INPUT_FILE.json"
OUTPUT_DIR = "OUTPUT_DIR"
PROMPTS_PATH = "ExperimentsScripts/prompts.json"

CATEGORY_FILTER = TARGET_CATEGORY  # set to None to process all rows

MODEL_RESPONSE_COL = f"model_response_{MODEL_NAME}"
MODEL_RESPONSE_SUBTYPE_COL = f"model_response_{MODEL_NAME}_subtype"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6
TOP_P = 0.9


def validate_config(model_name: str, target_category: str, shot_mode: str) -> None:
    if model_name not in {"Llama3", "Qwen25"}:
        raise ValueError("MODEL_NAME must be 'Llama3' or 'Qwen25'.")
    if target_category not in {"Migrant", "Frau"}:
        raise ValueError("TARGET_CATEGORY must be 'Migrant' or 'Frau'.")
    if shot_mode not in {"zeroshot", "fewshot"}:
        raise ValueError("SHOT_MODE must be 'zeroshot' or 'fewshot'.")


def derive_prompt_keys(target_category: str, shot_mode: str) -> tuple[str, str, str]:
    main_prompt_key = f"{target_category}_highlevel_{shot_mode}"
    solidarity_prompt_key = f"{target_category}_solidarity_{shot_mode}"
    antisolidarity_prompt_key = f"{target_category}_antisolidarity_{shot_mode}"
    return main_prompt_key, solidarity_prompt_key, antisolidarity_prompt_key


def derive_output_file_path(
    input_file: str,
    output_dir: str,
    target_category: str,
    model_name: str,
    shot_mode: str,
) -> str:
    input_path = Path(input_file)
    suffix = input_path.suffix.lower() or ".json"

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model_slug = model_name.lower()
    output_name = f"{target_category}_{model_slug}_{shot_mode}{suffix}"

    return str(output_dir_path / output_name)


def load_prompts(prompts_path: str) -> dict:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model():
    hf_token = os.getenv(HF_TOKEN_ENV)
    if not hf_token:
        raise EnvironmentError(f"{HF_TOKEN_ENV} environment variable is not set.")

    if MODEL_NAME == "Llama3":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        return pipeline(
            "text-generation",
            model=MODEL_ID,
            token=hf_token,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
            },
        )

    if MODEL_NAME == "Qwen25":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=hf_token,
            trust_remote_code=True,
            use_fast=True,
        )

        num_gpus = torch.cuda.device_count()
        max_memory = {i: "75GiB" for i in range(num_gpus)}
        max_memory["cpu"] = "120GiB"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=hf_token,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    raise ValueError(f"Unsupported MODEL_NAME '{MODEL_NAME}'.")


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


def generate_response(text_generator, instruction: str, text: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]

    prompt = text_generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    eos_token_id = text_generator.tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = text_generator.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    outputs = text_generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    generated_text = outputs[0]["generated_text"]
    if len(generated_text) > len(prompt):
        return generated_text[len(prompt):].strip()
    return ""


def normalize_sentences(field) -> list[str]:
    if isinstance(field, list):
        return [s.strip() for s in field if isinstance(s, str) and s.strip()]
    if isinstance(field, str):
        field = field.strip()
        return [field] if field else []
    return []


def build_full_text(row: pd.Series) -> str:
    prev_field = row.get("prev_sents", row.get("Previous", ""))
    sent_field = row.get("sent", row.get("Middle", ""))
    next_field = row.get("next_sents", row.get("Next", ""))

    prev_sents = normalize_sentences(prev_field)
    sent = str(sent_field).strip()
    next_sents = normalize_sentences(next_field)

    if not sent:
        raise ValueError("Missing middle sentence field in row.")

    return " ".join(prev_sents + [f"<MIDDLE>{sent}</MIDDLE>"] + next_sents)


def save_progress(df: pd.DataFrame, output_file: str, file_type: str) -> None:
    path = Path(output_file)

    if file_type == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        return

    if file_type == "csv":
        df.to_csv(path, index=False)
        return

    raise ValueError(f"Unsupported output format: {file_type}")


def load_input_data(file_path: str) -> tuple[pd.DataFrame, str]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f)), "json"

    if suffix == ".csv":
        return pd.read_csv(path), "csv"

    raise ValueError(f"Unsupported input format: {suffix}")


def row_matches_category(row: pd.Series, target_category: str | None) -> bool:
    if not target_category:
        return True

    row_category = row.get("category", row.get("Category", ""))
    row_category = str(row_category).strip().lower()
    return row_category == target_category.lower()


def already_processed(row: pd.Series) -> bool:
    high = row.get(MODEL_RESPONSE_COL)
    sub = row.get(MODEL_RESPONSE_SUBTYPE_COL)

    if not isinstance(high, str) or not high.strip():
        return False

    high_label = extract_high_level_label(high)

    if high_label in {"NONE", "MIXED"}:
        return True

    if high_label in {"SOLIDARITY", "ANTI-SOLIDARITY"}:
        return isinstance(sub, str) and bool(sub.strip())

    return False


def process_text(
    row: pd.Series,
    text_generator,
    prompts: dict,
    main_prompt_key: str,
    solidarity_prompt_key: str,
    antisolidarity_prompt_key: str,
) -> tuple[str, str | None]:
    if main_prompt_key not in prompts:
        raise KeyError(f"Prompt key '{main_prompt_key}' not found in {PROMPTS_PATH}")
    if solidarity_prompt_key not in prompts:
        raise KeyError(f"Prompt key '{solidarity_prompt_key}' not found in {PROMPTS_PATH}")
    if antisolidarity_prompt_key not in prompts:
        raise KeyError(f"Prompt key '{antisolidarity_prompt_key}' not found in {PROMPTS_PATH}")

    full_text = build_full_text(row)

    high_level_response = generate_response(
        text_generator=text_generator,
        instruction=prompts[main_prompt_key],
        text=full_text,
    )

    high_level_label = extract_high_level_label(high_level_response)
    subtype_response = None

    if high_level_label == "SOLIDARITY":
        subtype_response = generate_response(
            text_generator=text_generator,
            instruction=prompts[solidarity_prompt_key],
            text=full_text,
        )
    elif high_level_label == "ANTI-SOLIDARITY":
        subtype_response = generate_response(
            text_generator=text_generator,
            instruction=prompts[antisolidarity_prompt_key],
            text=full_text,
        )

    return high_level_response, subtype_response


def main():
    validate_config(MODEL_NAME, TARGET_CATEGORY, SHOT_MODE)

    input_path = Path(INPUT_FILE)
    output_file = derive_output_file_path(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        target_category=TARGET_CATEGORY,
        model_name=MODEL_NAME,
        shot_mode=SHOT_MODE,
    )
    output_path = Path(output_file)

    prompts = load_prompts(PROMPTS_PATH)
    text_generator = build_model()

    main_prompt_key, solidarity_prompt_key, antisolidarity_prompt_key = derive_prompt_keys(
        TARGET_CATEGORY,
        SHOT_MODE,
    )

    if output_path.exists():
        df, file_type = load_input_data(str(output_path))
        print(f"Resuming from existing output: {output_path}")
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        df, file_type = load_input_data(str(input_path))
        print(f"Starting from scratch using input: {input_path}")
        print(f"Output will be saved to: {output_path}")

    if MODEL_RESPONSE_COL not in df.columns:
        df[MODEL_RESPONSE_COL] = None

    if MODEL_RESPONSE_SUBTYPE_COL not in df.columns:
        df[MODEL_RESPONSE_SUBTYPE_COL] = None

    for index, row in df.iterrows():
        if not row_matches_category(row, CATEGORY_FILTER):
            continue

        if already_processed(row):
            continue

        try:
            high_response, subtype_response = process_text(
                row=row,
                text_generator=text_generator,
                prompts=prompts,
                main_prompt_key=main_prompt_key,
                solidarity_prompt_key=solidarity_prompt_key,
                antisolidarity_prompt_key=antisolidarity_prompt_key,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"Skipping index {index} due to CUDA OOM.")
            continue
        except Exception as e:
            print(f"Error at index {index}: {e}")
            continue

        df.at[index, MODEL_RESPONSE_COL] = high_response
        df.at[index, MODEL_RESPONSE_SUBTYPE_COL] = subtype_response

        save_progress(df, str(output_path), file_type)
        print(f"Saved progress after index {index}")

    print("Processing complete.")
    print(f"Final output: {output_path}")


if __name__ == "__main__":
    main()