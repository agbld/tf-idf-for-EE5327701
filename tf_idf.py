# python .\tf_idf_revised.py ../02_程式集/Coupang_Scraping-main/results -k 5 -i
# %%
# Import required libraries
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import jieba
import os
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from argparse import ArgumentParser

# Command line arguments
argparser = ArgumentParser()
argparser.add_argument('items_folder', type=str, help='Folder containing the items (csv files) to search.')
argparser.add_argument('-k', '--top_k', type=int, default=5, help='Number of top k items to return.')
argparser.add_argument('-f', '--file_idx', type=int, default=-1, help='File index of activate_item folder. Use -1 to load all files at once.')
argparser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode.')
argparser.add_argument('-a', '--all', action='store_true', help='Load all items without dropping duplicates.')
argparser.add_argument('-c', '--create', action='store_true', help='Create the TF-IDF models without using the saved models.')
argparser.add_argument('--api_server', action='store_true', help='Run in API server mode.')
args = argparser.parse_args()
items_folder = args.items_folder
top_k = args.top_k
file_idx = args.file_idx
interactive = args.interactive
drop_duplicates = not args.all
create = args.create

# %%

# Define variables directly instead of using ArgumentParser
# 以下是在 ipynb 測試時，我用變數指定替代 argparser 的功用
# items_folder = '../02_程式集/Coupang_Scraping-main/results'
# top_k = 5
# file_idx = -1
# interactive = True
# drop_duplicates = True
# create = False


# %%

# Check if the items folder exists
if not os.path.exists(items_folder):
    print(f'Error: Folder "{items_folder}" not found.')
    # Create the folder if it doesn't exist in a non-interactive way
    os.makedirs(items_folder)
    print(f'Folder "{items_folder}" created. Please add the items (csv files) to this folder and rerun the script.')
else:
    if file_idx == -1:
        print(f'Loading all files from: "{items_folder}"')
    else:
        print(f'Loading {file_idx}th file from: "{items_folder}"')


# %%
# 設定 pickle 檔案路徑
pickle_file = 'items_data.pkl'

# 檢查 pickle 檔案是否存在
if os.path.exists(pickle_file):
    # 如果 pickle 檔案存在，直接加載
    print(f'Loading data from {pickle_file}...')
    with open(pickle_file, 'rb') as f:
        items_df = pickle.load(f)
    print(f'Loaded data from {pickle_file}. Total items: {len(items_df)}')
else:
    # 如果 pickle 檔案不存在，重新合併 CSV 檔案
    print(f'{pickle_file} not found. Merging CSV files...')
    timer_start = time.time()

    # 根據 file_idx 加載文件
    if file_idx >= 0:
        path_to_item_file = [file for file in os.listdir(items_folder) if file.endswith('.csv')][file_idx]
        items_df = pd.read_csv(os.path.join(items_folder, path_to_item_file), usecols=['product_name'])
    else:
        path_to_item_files = [file for file in os.listdir(items_folder) if file.endswith('.csv')]
        items_df = []
        for file in path_to_item_files:
            try:
                items_df.append(pd.read_csv(os.path.join(items_folder, file), usecols=['product_name']))
            except:
                print(f'Error loading file: {file}')
        print(f'Loaded {len(items_df)} files.')
        items_df = pd.concat(items_df, ignore_index=True)

    # 預處理 product_name 欄位
    items_df['product_name'] = items_df['product_name'].map(html.unescape)
    items_df['product_name'] = items_df['product_name'].fillna('')

    if drop_duplicates:
        items_df = items_df.drop_duplicates(subset='product_name')

    print(f'Processed {len(items_df)} items in {time.time() - timer_start:.2f} seconds.')

    # 保存為 pickle 檔案
    with open(pickle_file, 'wb') as f:
        pickle.dump(items_df, f)
    print(f'Data saved to {pickle_file}.')


# %%
# Initialize tokenizer
timer_start = time.time()

# 定義簡單的分詞器
def jieba_tokenizer(text):
    # 使用 jieba 的精確模式進行分詞
    tokens = jieba.lcut(text, cut_all=False)
    # 定義不需要的符號列表
    stop_words = ['【','】','/','~','＊','、','（','）','+','‧',' ','']
    # 過濾掉不需要的符號
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# 使用 jieba_tokenizer 進行分詞
tokenizer = jieba_tokenizer


# %%

# Path to save/load the models
model_path = 'tf_idf_checkpoint.pkl'

# Function to save the models
def save_models_and_matrices(tfidf, items_tfidf_matrix, path):
    with open(path, 'wb') as file:
        pickle.dump({
            'tfidf': tfidf,
            'items_tfidf_matrix': items_tfidf_matrix,
        }, file)

# Function to load the models
def load_models_and_matrices(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['tfidf'], data['items_tfidf_matrix']


# %%

# Check if the models are already saved
if os.path.exists(model_path) and not create:
    # If saved, load the models
    tfidf, items_tfidf_matrix = load_models_and_matrices(model_path)
else:
    # If not saved, create the models
    print('TF-IDF models not found. Creating them...')

    tfidf = TfidfVectorizer(token_pattern=None, tokenizer=tokenizer, ngram_range=(1,2))
    items_tfidf_matrix = tfidf.fit_transform(tqdm(items_df['product_name']))
    
    save_models_and_matrices(tfidf, items_tfidf_matrix, model_path)

print(f'TF-IDF models loaded in {time.time() - timer_start:.2f} seconds.')


# %%

# Function to search for the top k items
def search(query):
    query_tfidf = tfidf.transform([query]) # sparse array
    scores = cosine_similarity(query_tfidf, items_tfidf_matrix)
    top_k_indices = np.argsort(-scores[0])[:top_k]
    
    top_k_names = items_df['product_name'].values[top_k_indices]
    top_k_scores = scores[0][top_k_indices]

    return top_k_names, top_k_scores


# Run in interactive mode
if interactive and not args.api_server:

    while True:
        query = input('Enter query: ')
        if query == 'exit':
            break
        top_k_names, scores = search(query)

        for i, name in enumerate(top_k_names):
            print(f'[Rank {i+1} ({round(scores[i], 4)})] {name}')

# %%



