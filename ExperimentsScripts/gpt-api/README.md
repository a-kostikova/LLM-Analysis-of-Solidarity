## GPT inference script

The GPT inference script runs prompting-based classification with OpenAI chat models on `.json` or `.csv` datasets.

### Configuration

The script is configured through constants at the top of the file:

- `MODEL_NAME`: OpenAI model name, for example `gpt-4-1106-preview`, `gpt-3.5-turbo-0125`, `gpt-5-chat`, or `gpt-5-chat-latest`
- `TARGET_CATEGORY`: target category, either `Migrant` or `Frau`
- `TASK_MODE`: prompting setup, either `one_step` or `two_step`
- `SHOT_MODE`: prompt type, either `zeroshot` or `fewshot`
- `PROMPTS_PATH`: path to `prompts.json`
- `INPUT_FILE_PATH`: path to the input dataset

### Prompting modes

Prompt keys are derived automatically from `TARGET_CATEGORY`, `TASK_MODE`, and `SHOT_MODE`.

- `one_step` uses a single combined prompt:
  - `Migrant_zeroshot`, `Migrant_fewshot`
  - `Frau_zeroshot`, `Frau_fewshot`

- `two_step` uses:
  - a high-level prompt (`*_highlevel_*`)
  - followed, if needed, by a subtype prompt (`*_solidarity_*` or `*_antisolidarity_*`)

Subtype prompting is only run if the high-level response is `SOLIDARITY` or `ANTI-SOLIDARITY`.

### Input Formats

Supported formats:

- **CSV**: `Previous`, `Middle`, `Next`, `Category`
- **JSON**: `prev_sents`, `sent`, `next_sents`, `category`

The prompt text is built from previous, middle, and next sentences, with the middle sentence marked as `<MIDDLE>...</MIDDLE>`. For `one_step`, no middle tags are added.

### Processing behavior

- Only rows matching `TARGET_CATEGORY` are processed
- Already processed rows are skipped
- Progress is saved every 10 rows
- Output is saved again after all rows are processed

For `one_step`, one response column is used (`model_response_{model_name}`).  
For `two_step`, both a high-level (`model_response_{model_name}`) and a subtype response column (`model_response_{model_name}_subtype`) are used.

Final project labels are extracted from these `model_response_*` columns with [`ExtractLabel.py`](../ExtractLabel.py).

### Output file naming

The output filename is derived automatically from:

- `TARGET_CATEGORY`
- `MODEL_NAME`
- `SHOT_MODE`

Examples:

- `Migrant_gpt4_zeroshot.json`
- `Frau_gpt35_fewshot.json`
- `Migrant_gpt5_zeroshot.json`

The file is saved in the same directory as the input dataset.

### Requirements

The script requires an OpenAI API key in the environment:

```bash
export OPENAI_API_KEY=...
```