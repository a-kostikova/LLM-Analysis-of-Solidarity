# Experiment Scripts

This folder contains the scripts used to run the experiments reported in the project.

The subfolders are organized by experimental setup:

- 📂 [`bert/`](./bert/): scripts for the hierarchical BERT baseline
- 📂 [`sbert/`](./sbert/): scripts for the hierarchical SBERT baseline
- 📂 [`gpt-api/`](./gpt-api/): scripts for OpenAI API-based prompting experiments
- 📂 [`llama-cpp/`](./llama-cpp/): scripts for local inference experiments with GGUF models via `llama.cpp`
- 📂 [`pipeline/`](./pipeline/): scripts for inference experiments with Hugging Face generation pipelines
- 📄 [`ExtractLabel.py`](./ExtractLabel.py): helper script for extracting final labels from raw model outputs
- 📄 [`prompts.json`](./prompts.json): prompt templates used for prompting-based experiments

## Utility scripts

### [`prompts.json`](./prompts.json)

This file contains the prompt templates used by the prompting-based experiments, including `one_step` and `two_step` setups. See [`gpt-api/README.md`](./gpt-api/README.md) for an explanation of these setups.

### [`ExtractLabel.py`](./ExtractLabel.py)

This script extracts final labels from raw model outputs and converts them to the label scheme used in this project (see the [Label scheme](#label-scheme)).

It is configured through constants at the top of the file:

- `INPUT_FILE`: path to the prediction file (`.json` or `.csv`)
- `MODEL_NAME`: model suffix used in response column names, for example `GPT4`, `GPT5`, `Llama3`, `Qwen25`, etc.
- `TASK_MODE`: either `one_step` or `two_step`

The script:

- reads a prediction file in `.json` or `.csv` format
- reads model response columns of the form:
  - `model_response_{MODEL_NAME}`
  - and, for `two_step`, `model_response_{MODEL_NAME}_subtype`
- extracts labels from generated text
- maps long-form labels such as `GROUP-BASED SOLIDARITY` to project labels such as `s.group-based` (see [Label scheme](#label-scheme))
- writes the extracted labels back into the same file in:
  - `extracted_label_{MODEL_NAME}`

Examples:

- `MODEL_NAME="GPT4", TASK_MODE="one_step"`
  - reads `model_response_GPT4`
  - writes `extracted_label_GPT4`

- `MODEL_NAME="Qwen25", TASK_MODE="two_step"`
  - reads `model_response_Qwen25` and `model_response_Qwen25_subtype`
  - writes `extracted_label_Qwen25`

## Label scheme

The experiments use the following labels:

| Label               | Description                     |
|---------------------|---------------------------------|
| `s.group-based`     | solidarity, group-based         |
| `s.exchange-based`  | solidarity, exchange-based      |
| `s.compassionate`   | solidarity, compassionate       |
| `s.empathic`        | solidarity, empathic            |
| `s.none`            | solidarity, no subtype          |
| `as.group-based`    | anti-solidarity, group-based    |
| `as.exchange-based` | anti-solidarity, exchange-based |
| `as.compassionate`  | anti-solidarity, compassionate  |
| `as.empathic`       | anti-solidarity, empathic       |
| `as.none`           | anti-solidarity, no subtype     |
| `mixed.none`        | mixed stance                    |
| `none.none`         | none                            |