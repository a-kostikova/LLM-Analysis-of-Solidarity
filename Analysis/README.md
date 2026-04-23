# Design-Based Supervised Learning (DSL) Analysis

This folder contains the code and analysis files used to apply **Design-Based Supervised Learning (DSL)** to parliamentary debates.

Our implementation is based on the DSL framework introduced in:

- Egami, Naoki, Musashi Hinck, Brandon M. Stewart, and Hanying Wei. *Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models.* NeurIPS 2023. [Paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d862f7f5445255090de13b825b880d59-Abstract-Conference.html)

In our setting, DSL is used to combine:
- soft-label predictions from a committee of LLMs, and
- soft-label human annotations derived from multiple annotators for a smaller subset of instances,

in order to construct bias-corrected outputs for downstream analysis of long-term trends in parliamentary migration discourse.

The implementation provided here applies the DSL workflow to multi-class soft-label distributions for:
- **high-level labels** (`solidarity`, `anti-solidarity`, `mixed`, `none`), and
- **fine-grained labels** (solidarity and anti-solidarity subtypes, plus `mixed.none` and `none.none`).

## Folder structure

- 📓 [`dsl_pipeline.ipynb`](./dsl_pipeline.ipynb): notebook for constructing DSL inputs, running DSL, and saving output arrays
- 🐍 [`multi_label_dsl.py`](./multi_label_dsl.py): DSL implementation used in the project
- 📁 [`dsl_outputs/`](./dsl_outputs): saved NumPy arrays produced by the DSL pipeline
- 📁 [`Plotting/`](./Plotting): example plotting scripts for aggregating DSL outputs

## Input requirements

The DSL notebooks expect an input JSON file that contains:
- document-level metadata such as `id`, `year`, and other grouping variables
- model-predicted labels
- human annotator labels for the subset of instances used as gold-standard data

Before running the DSL pipeline, the human annotation files from [`Data/HumanAnnotatedDataset`](./Data/HumanAnnotatedDataset) must be merged into the model-prediction file [`Data/ModelPredictedData/Migrant_1867-2025_AllModels.json`](./Data/ModelPredictedData/Migrant_1867-2025_AllModels.json).

## DSL pipeline outputs

The notebook saves the following `.npy` files in [`dsl_outputs/`](./dsl_outputs):

- `Ytilde_full_{scheme}.npy`: DSL-adjusted class scores for all retained samples
- `G_full_{scheme}.npy`: cross-fitted first-stage predictions for all retained samples
- `ids_full_{scheme}.npy`: sample IDs aligned row-wise with `Ytilde` and `G`

Here `{scheme}` is either:
- `high` for the high-level 4-label setup
- `fine` for the fine-grained 10-label setup

## Notes on plotting

The saved DSL arrays do not include full document metadata such as year, party, or keyword. Therefore, the plotting scripts use three inputs together:

- the full JSON file, to recover metadata such as `id`, `year`, `party`, or `keyword`
- `Ytilde_full_{scheme}.npy`, which contains the DSL-adjusted class scores
- `ids_full_{scheme}.npy`, which maps rows in the DSL arrays back to document IDs

This allows the DSL outputs to be merged back to the original metadata and aggregated by decade or other grouping variables.