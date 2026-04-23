## Prompt-based Inference with Transformers Pipeline

[`pipeline-inference.py`](./pipeline-inference.py) runs prompt-based inference with:

- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`

The script performs two-step classification for `Migrant` and `Frau`: first a high-level label, then a subtype label for `SOLIDARITY` or `ANTI-SOLIDARITY`.

### Configuration

The script is configured through constants at the top of the file:

- `MODEL_NAME`: `Llama3` or `Qwen25`
- `MODEL_ID`: Hugging Face model ID
- `HF_TOKEN_ENV`: environment variable containing the Hugging Face token
- `TARGET_CATEGORY`: `Migrant` or `Frau`
- `SHOT_MODE`: `zeroshot` or `fewshot`
- `INPUT_FILE`: input file (`.json` or `.csv`)
- `OUTPUT_DIR`: output directory
- `PROMPTS_PATH`: path to `prompts.json`
- `CATEGORY_FILTER`: category filter; set to `None` to process all rows
- `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`: generation settings

The output filename is generated automatically as:

`{TARGET_CATEGORY}_{model_name_lower}_{SHOT_MODE}.{ext}`

### Prompt keys

Prompt keys are derived from `TARGET_CATEGORY` and `SHOT_MODE` using [`prompts.json`](../prompts.json):

- `{TARGET_CATEGORY}_highlevel_{SHOT_MODE}`
- `{TARGET_CATEGORY}_solidarity_{SHOT_MODE}`
- `{TARGET_CATEGORY}_antisolidarity_{SHOT_MODE}`

### Input

Supported formats:

- **CSV**: `Previous`, `Middle`, `Next`, `Category`
- **JSON**: `prev_sents`, `sent`, `next_sents`, `category`

The prompt text is built from previous, middle, and next sentences, with the middle sentence marked as `<MIDDLE>...</MIDDLE>`.

### Processing

The script:

1. predicts a high-level label: `SOLIDARITY`, `ANTI-SOLIDARITY`, `MIXED`, or `NONE`
2. runs a subtype prompt only for `SOLIDARITY` and `ANTI-SOLIDARITY`

It also:

- processes only rows matching `CATEGORY_FILTER`
- resumes from an existing output file
- skips already processed rows
- saves progress after each processed row
- skips rows that trigger CUDA out-of-memory errors

### Model loading

- `Llama3`: 4-bit quantization via `transformers.pipeline`
- `Qwen25`: 8-bit quantization with explicit tokenizer/model loading and `device_map="auto"`

### Output

The script writes raw model responses to:

- `model_response_{MODEL_NAME}`
- `model_response_{MODEL_NAME}_subtype`

Final labels can be extracted later with [`../ExtractLabel.py`](../ExtractLabel.py).

### Requirements

Tested with Python 3.10.19 and the package versions listed in `requirements.txt`.

The script requires access to the selected Hugging Face model and a Hugging Face token:

```bash
export HF_TOKEN=...
```