import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import jieba
import os
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from argparse import ArgumentParser
import pickle

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

# Check if the items folder exists
if not os.path.exists(items_folder):
    print(f'Error: Folder "{items_folder}" not found.')
    response = input(f'Do you want to create the folder "{items_folder}"? (yes/no): ').strip().lower()
    if response == 'yes':
        os.makedirs(items_folder)
        print(f'Folder "{items_folder}" created. Please add the items (csv files) to this folder and rerun the script.')
    else:
        print('Please provide the correct path to the items folder and rerun the script.')
    exit()

if file_idx == -1:
    print(f'Loading all files from: "{items_folder}"')
else:
    print(f'Loading {file_idx}th file from: "{items_folder}"')

# load item file from activate_item folder by file_idx
timer_start = time.time()
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
    path_to_item_file = 'all'

# preprocess item_df
items_df['product_name'] = items_df['product_name'].map(html.unescape)
items_df['product_name'] = items_df['product_name'].fillna('')

if drop_duplicates:
    items_df = items_df.drop_duplicates(subset='product_name')
print(f'Processed {len(items_df)} items in {time.time() - timer_start:.2f} seconds.')

timer_start = time.time()

# Disable jieba cache logging
jieba.setLogLevel(jieba.logging.WARN)
class JiebaTokenizer(object):
    def __init__(self, class_name):
        self.class_name = class_name
        for each_name in self.class_name:
            userdict_path = './Lexicon_merge/{}.txt'.format(each_name)
            if os.path.exists(userdict_path):
                jieba.load_userdict(userdict_path)
            else:
                print(f"User dictionary {userdict_path} not found, skipping.")
    
    def __call__(self, x):
        tokens = jieba.lcut(x, cut_all=False)
        stop_words = ['【','】','/','~','＊','、','（','）','+','‧',' ','']
        tokens = [t for t in tokens if t not in stop_words]
        return tokens

tokenizer = JiebaTokenizer(class_name=['type','brand','p-other'])

# Path to save/load the models
model_path = 'tf_idf_checkpoint.pkl'

# Function to save the models
def save_models_and_matrices(tfidf, items_tfidf_matrix, tfidf_char, items_tfidf_matrix_char, path):
    with open(path, 'wb') as file:
        pickle.dump({
            'tfidf': tfidf,
            'items_tfidf_matrix': items_tfidf_matrix,
            'tfidf_char': tfidf_char,
            'items_tfidf_matrix_char': items_tfidf_matrix_char
        }, file)

# Function to load the models
def load_models_and_matrices(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['tfidf'], data['items_tfidf_matrix'], data['tfidf_char'], data['items_tfidf_matrix_char']

# Check if the models are already saved
if os.path.exists(model_path) and not create:
    # If saved, load the models
    tfidf, items_tfidf_matrix, tfidf_char, items_tfidf_matrix_char = load_models_and_matrices(model_path)
else:
    # If not saved, create the models
    print('TF-IDF models not found. Creating them...')

    tfidf = TfidfVectorizer(token_pattern=None, tokenizer=tokenizer, ngram_range=(1,2))
    items_tfidf_matrix = tfidf.fit_transform(tqdm(items_df['product_name']))
    
    tfidf_char = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", analyzer='char')
    items_tfidf_matrix_char = tfidf_char.fit_transform(items_df['product_name'])

    save_models_and_matrices(tfidf, items_tfidf_matrix, tfidf_char, items_tfidf_matrix_char, model_path)

print(f'TF-IDF models loaded in {time.time() - timer_start:.2f} seconds.')

# Function to search for the top k items
def search(query):
    query_tfidf = tfidf.transform([query]) # sparse array
    scores = cosine_similarity(query_tfidf, items_tfidf_matrix)
    top_k_indices = np.argsort(-scores[0])[:top_k]
    sum_of_score = sum(scores[0])
    
    if sum_of_score < 10 : 
        query_tfidf = tfidf_char.transform([query]) # sparse array
        scores = cosine_similarity(query_tfidf, items_tfidf_matrix_char)
        top_k_indices = np.argsort(-scores[0])[:top_k]
        sum_of_score = sum(scores[0])
        
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

# Run in API server mode. 
# Note: This part is not necessary to run if you are student. It is not required in the assignment.
elif args.api_server:
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route('/search', methods=['GET'])
    def search_api():
        query = request.args.get('query')
        top_k_names, scores = search(query)
        return jsonify({'top_k_names': top_k_names.tolist(), 'scores': scores.tolist()})
    
    app.run(host='0.0.0.0', port=5000)

    # Example usage: http://localhost:5000/search?query=iphone