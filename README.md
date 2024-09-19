# TF-IDF Search System for NTUST Big Data Analysis Course (EE5327701)

This repository contains a command-line based search system that utilizes TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to retrieve the most relevant items based on a query. It is designed as a demonstration for the **NTUST Big Data Analysis course (EE5327701)**.

## Features
- **Efficient Text Search:** Uses TF-IDF vectorization and cosine similarity to return the top-k most similar items to a user query.
- **Jieba Tokenization:** Supports Chinese tokenization using Jieba with custom dictionaries.
- **Interactive Mode:** Allows users to search interactively for items and returns similarity scores.
- **CSV File Handling:** Processes multiple CSV files containing product information.

## Table of Contents
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Command-line Arguments](#command-line-arguments)
- [Interactive Mode](#interactive-mode)
- [Notes](#notes)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/agbld/tf-idf-for-EE5327701.git
   cd tf-idf-for-EE5327701
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The required dependencies include:
   - `tqdm`
   - `numpy`
   - `pandas`
   - `jieba`
   - `scikit-learn`

## Setup

Before running the script, ensure that:
1. Your data (CSV files) is placed in the `items_folder` as specified in the command-line arguments (default: `./items`). Each CSV should contain a `product_name` column.
2. Optional Jieba dictionaries (for tokenizing specific domains) are located in the `Lexicon_merge` directory.

## Usage

The program can be run from the command line with various options. Below is a basic usage example:

```bash
python tf_idf.py ./items -k 5 -ic
```

This will load all CSV files from the `./items` folder, use the top 5 results, and run the program in interactive mode.

## Command-line Arguments

The script accepts the following command-line arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `items_folder` | `str` | `./items` | Folder containing the items (CSV files) to search. |
| `-k`, `--top_k` | `int` | `5` | Number of top k items to return. |
| `-f`, `--file_idx` | `int` | `-1` | File index of the item folder. Use `-1` to load all files at once. |
| `-i`, `--interactive` | `flag` | `False` | Run the system in interactive mode for query input. |
| `-s`, `--sample_size` | `int` | `100000` | Number of items to sample from the dataset for TF-IDF model creation. Use -1 to load all items. |
| `-a`, `--all` | `flag` | `False` | Load all items without dropping duplicates. |
| `-c`, `--create` | `flag` | `False` | Create the TF-IDF models from scratch without using saved models. |

### Example Usages

1. **Load all CSV files and return top 5 results:**
   ```bash
   python tf_idf.py ./items -k 5
   ```

2. **Load a specific CSV file by index (e.g., 2nd file):**
   ```bash
   python tf_idf.py ./items -f 2
   ```

3. **Interactive mode for live querying:**
   ```bash
   python tf_idf.py ./items -i
   ```

4. **Force creation of new TF-IDF models:**
   ```bash
   python tf_idf.py ./items -c
   ```

5. **Load all items without dropping duplicates:**
   ```bash
   python tf_idf.py ./items -a -s -1
   ```

## Interactive Mode

In interactive mode, the program allows users to input queries and return the top-k most similar product names based on TF-IDF cosine similarity. To enter interactive mode, use the `-i` flag.

- To exit the interactive session, type `exit`.
- Example interaction:

   ```bash
   Enter query: 美髮
   [Rank 1 (0.287)] APLB 損傷髮質護髮洗髮露, 500ml, 1瓶
   [Rank 2 (0.2715)] moremo 咖啡因精華強健髮根洗髮精 油性髮適用, 500ml, 1瓶
   [Rank 3 (0.2708)] ReEn 強健髮根洗髮精 油性髮質適用, 950ml, 1瓶
   [Rank 4 (0.2606)] 長髮公主的秘密 魔髮香氛控油洗髮精, 1L, 1瓶
   [Rank 5 (0.2383)] BRO&T!PS 健髮洗髮精 油性髮質, 1入, 500ml
   Enter query: exit
   ```

## Notes

- The system supports Chinese text tokenization using Jieba. If you want to use custom dictionaries, place them in the `Lexicon_merge` folder and ensure the filenames match the specified categories (`type`, `brand`, `p-other`).
- TF-IDF models are saved in `tf_idf_checkpoint.pkl` and loaded if they already exist, unless the `-c` flag is used to recreate them.

---

This repository is created for academic purposes as part of the NTUST Big Data Analysis course (EE5327701). Feel free to modify and extend it for other projects!