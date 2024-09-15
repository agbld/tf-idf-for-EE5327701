# Product Search Engine using TF-IDF with Jieba Tokenization

This repository implements a product search engine based on TF-IDF vectorization, incorporating tokenization through Jieba (a popular Chinese text segmentation library). The project allows for efficient searching of product names, based on both word-level and character-level TF-IDF vectorization techniques.

## Features
- Tokenization of product names using custom dictionaries and Jieba.
- TF-IDF vectorization of the tokenized product names.
- Cosine similarity-based search to find the top matching products.
- Handles both word and character-level TF-IDF searches to improve query matching.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Setup

1. **Prepare Product Files**: Ensure you have product data files in CSV format in the specified folder (default: `./results`).
   - These files should contain a column named `product_name`.

2. **Custom Dictionaries**: 
   - Place custom dictionaries for Jieba tokenization in the `./Lexicon_merge/` folder (e.g., `type.txt`, `brand.txt`, `p-other.txt`).
   - If any of these files are missing, Jieba will skip loading that dictionary and continue processing.

## Usage

You can run the script from the command line, specifying the dataset and parameters. For example:

```bash
python search_script.py --items_folder ./results --file_idx 0 --top_k 50 --N_ROWS 1000
```

- `--items_folder`: The folder containing product CSV files.
- `--file_idx`: Index of the file to load (use `-1` to load all files).
- `--top_k`: Number of top matching products to return (default: 50).
- `--N_ROWS`: Optionally limit the number of rows to load (default: None).

## Search Functionality

After setting up, you can perform a product search using the following function:

```python
top_k_names = search("your query", top_k=10)
```

This will return the top K matching product names based on the input query.

### How it works:
1. The product names are vectorized using TF-IDF.
2. A cosine similarity score is calculated between the query and product names.
3. The top `K` products with the highest similarity scores are returned.

## Saving and Loading Models
The script automatically saves the TF-IDF models and product matrices for future use to avoid re-computation. These are saved as:
- `tfidf_model.pkl`
- `tfidf_char_model.pkl`
- `items_tfidf_matrix.npz`
- `items_tfidf_matrix_char.npz`

If the files already exist, the script will load them from disk, making the search process faster.

## License
This project is open-sourced under the MIT License.