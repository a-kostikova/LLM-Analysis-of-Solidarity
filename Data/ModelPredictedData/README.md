# Model-Predicted Data

This folder contains model-predicted labels for the `Migrant` dataset.

The full collection includes all 63,000 instances annotated with the following models:

- `gptoss120b` (gpt-oss-120B)
- `llama3` (Llama-3.3-70B)
- `qwen25` (Qwen-2.5-72B)

In addition, a sampled subset of 18,300 migrant instances also contains `gpt4` (GPT-4) annotations. This subset corresponds to the proportional sample released with ["Fine-Grained Detection of Solidarity for Women and Migrants in 155 Years of German Parliamentary Debates"](https://aclanthology.org/2024.emnlp-main.337/) and the associated [github repo (DominikBeese/FairGer)](https://github.com/DominikBeese/FairGer).

## Instance format

Each instance contains the following keys:

| Key          | Value  | Description                                        |
|--------------|--------|----------------------------------------------------|
| `id`         | string | distinct sentence id                               |
| `era`        | string | `rt` for Reichstag or `bt` for Bundestag           |
| `type`       | string | only for Reichstag data                            |
| `period`     | number | election period                                    |
| `no`         | number | number of the sitting                              |
| `line`       | number | line number of the sentence                        |
| `year`       | number | year of the sitting                                |
| `month`      | number | month of the sitting                               |
| `day`        | number | day of the sitting                                 |
| `category`   | string | `Migrant`                                          |
| `keyword`    | string | keyword contained in the target sentence           |
| `party`      | string | party affiliation, where available; Bundestag only |
| `prev_sents` | array  | the three preceding sentences                      |
| `sent`       | string | target sentence                                    |
| `next_sents` | array  | the three following sentences                      |
| `models`     | object | nested model predictions and raw model outputs     |

## Model predictions

Model predictions are stored in the `models` field. Each model is represented as a nested object, for example:

```json
"models": {
    "llama3": {
        "extracted_label": "...",
        "highlevel_raw_response": "...",
        "subtype_raw_response"
    },
    "qwen25": {
        "extracted_label": "...",
        "highlevel_raw_response": "...",
        "subtype_raw_response": "...",
    },
    "gptoss120b": {
        "extracted_label": "...",
        "highlevel_raw_response": "...",
        "subtype_raw_response": "...",
    },
    "gpt4": {
        "extracted_label": "...",
        "raw_response": "...",
        }
}
```

During inference, the models were first prompted to predict a high-level label (`SOLIDARITY`, `ANTI-SOLIDARITY`, `MIXED`, or `NONE`). The raw response to this prompt is stored in `highlevel_raw_response`.
If the model selected `SOLIDARITY` or `ANTI-SOLIDARITY`, it was then prompted to predict a subtype (`GROUP-BASED`, `EXCHANGE-BASED`, `COMPASSIONATE`, or `EMPATHIC`). The raw response to this second prompt is stored in `subtype_raw_response`.
GPT-4 predictions are stored slightly differently. They include `raw_response` and `extracted_label`, but no separate `subtype_raw_response` field, because the high-level and subtype prediction were combined in a single step.

The field `extracted_label` stores the final project label extracted from the raw model response. It uses the following values:

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
