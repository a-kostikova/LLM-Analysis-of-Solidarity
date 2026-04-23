# Human Annotated Data

This folder contains two related human-annotated datasets on solidarity in German parliamentary debates.

The first dataset is the human-annotated data (2,864 instances) released alongside the paper [Fine-Grained Detection of Solidarity for Women and Migrants in 155 Years of German Parliamentary Debates](https://aclanthology.org/2024.emnlp-main.337/#) ([DominikBeese/FairGer](https://github.com/DominikBeese/FairGer)), extended to include **all annotators' labels**, not only the consensus labels.

The second dataset is the extension (756 instances) introduced in [LLM Analysis of 150+ Years of German Parliamentary Debates on Migration Reveals a Shift from Post-War Solidarity to Anti-Solidarity in the Last Decade](https://arxiv.org/pdf/2509.07274).

## Folder structure

### Original human-annotated dataset

The root of this folder contains the earlier human-annotated dataset, now provided with all annotation inputs and encrypted label files for both target groups:

- 📄 [`HumanAnnotatedDatasetInputs_Frau.json`](./HumanAnnotatedDatasetInputs_Frau.json): input instances for the woman dataset
- 📄 [`HumanAnnotatedDatasetInputs_Migrant.json`](./HumanAnnotatedDatasetInputs_Migrant.json): input instances for the migrant dataset
- 🔒 [`AllAnnotationLabels_Frau.json.gpg`](./AllAnnotationLabels_Frau.json.gpg): encrypted annotation labels for the woman dataset; see [Decrypting the label files](#decrypting-the-label-files)
- 🔒 [`AllAnnotationLabels_Migrant.json.gpg`](./AllAnnotationLabels_Migrant.json.gpg): encrypted annotation labels for the migrant dataset; see [Decrypting the label files](#decrypting-the-label-files)

### Extension dataset

The folder [`EXTENSION_HumanAnnotatedDataset`](./EXTENSION_HumanAnnotatedDataset) contains the extension dataset introduced later:

- 📄 [`EXTENSION_HumanAnnotatedDatasetInputs_Migrant.json`](./EXTENSION_HumanAnnotatedDataset/EXTENSION_HumanAnnotatedDatasetInputs_Migrant.json): input instances for the extension migrant dataset
- 🔒 [`EXTENSION_AllAnnotationLabels_Migrant.json.gpg`](./EXTENSION_HumanAnnotatedDataset/EXTENSION_AllAnnotationLabels_Migrant.json.gpg): encrypted annotation labels for the extension migrant dataset; see [Decrypting the label files](#decrypting-the-label-files)

## Input file format

Each input JSON file contains a list of sentences with the following keys:

| Key               | Value  | Description                                  |
|-------------------|--------|----------------------------------------------|
| `id`              | string | distinct id of the instance                  |
| `era`             | string | `rt` for Reichstag or `bt` for Bundestag     |
| `type`            | string | only for Reichstag data                      |
| `period`          | number | election period                              |
| `no`              | number | number of the sitting                        |
| `line`            | number | line number of the sentence                  |
| `year`            | number | year of the sitting                          |
| `month`           | number | month of the sitting                         |
| `day`             | number | day of the sitting                           |
| `category`        | string | `Frau` for woman or `Migrant` for migrant    |
| `keyword`         | string | keyword contained in the target sentence     |
| `prev_sents`      | array  | the three preceding sentences                |
| `sent`            | string | target sentence                              |
| `next_sents`      | array  | the three following sentences                |
| `consensus_level` | string | `curated`, `majority`, or `single`           |

## Label file format

Each encrypted label JSON file contains annotation labels keyed by instance id.

| Key            | Value  | Description                        |
|----------------|--------|------------------------------------|
| `id`           | string | distinct id of the instance        |
| `annotator{n}` | string | label assigned by annotator `n`    |
| `label`        | string | consensus label for the instance   |

The `label` field uses the following values:

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

## Decrypting the label files

The label files are distributed in encrypted form (`.gpg`).

To request access to the decryption password, please contact [aida.kostikova@uni-bielefeld.de](mailto:aida.kostikova@uni-bielefeld.de) or [lovegoodaida@gmail.com](mailto:lovegoodaida@gmail.com). The password is shared separately to reduce benchmark contamination risk.

After obtaining the password, you can decrypt the files with:

```bash
gpg -o AllAnnotationLabels_Frau.json -d AllAnnotationLabels_Frau.json.gpg
gpg -o AllAnnotationLabels_Migrant.json -d AllAnnotationLabels_Migrant.json.gpg
gpg -o EXTENSION_AllAnnotationLabels_Migrant.json -d EXTENSION_HumanAnnotatedDataset/EXTENSION_AllAnnotationLabels_Migrant.json.gpg
```

## Benchmark protection

Please do not:
- redistribute decrypted label files publicly
- use protected labels for model training
- upload protected benchmark content to closed APIs without training-exclusion guarantees

This release follows the benchmark-protection rationale discussed in [Jacovi et al. (2023)](https://aclanthology.org/2023.emnlp-main.308/).

## License

The original contributions of the dataset authors, such as labels and annotations, are licensed under the Creative Commons Attribution-NoDerivatives 4.0 International License (CC BY-ND 4.0).

This license does not apply to underlying third-party materials or official-source texts, including Bundestag/Reichstag source texts, which remain subject to their own legal status and source-attribution requirements.
