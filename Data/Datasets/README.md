# Datasets
This folder contains our Woman and Migrant datasets which were updated to include instances up to June 2025.
 * 🗃 [`Woman` dataset](Frau_1867-2025.json)
 * 🗃 [`Migrant` dataset](Migrant_1867-2025.json)

Json files contain a list of sentences with the following keys:

| Key          | Value  | Description                                                |
|--------------|--------|------------------------------------------------------------|
| `id`         | string | distinct id of the sentence                                |
| `era`        | string | `rt` for Reichstag _or_ `bt` for Bundestag                 |
| `type`       | string | only for Reichstag data, see table above                   |
| `period`     | number | election period                                            |
| `no`         | number | number of the sitting                                      |
| `line`       | number | line number of the sentence                                |
| `year`       | number | year of the sitting                                        |
| `month`      | number | month of the sitting                                       |
| `day`        | number | day of the sitting                                         |
| `category`   | string | `Frau` for woman _or_ `Migrant` for migrant                |
| `keyword`    | string | the keyword contained in the target sentence               |
| `prev_sents` | array  | array of the preceding three sentences                     |
| `sent`       | string | target sentence                                            |
| `next_sents` | array  | array of the following three sentences                     |
| `party`      | string | party affiliation, where available; used only in `Migrant` |

**Note:** 

The `party` field is available for some instances in the `Migrant` dataset (49k out of 63k) and only for Bundestag entries.