#%%
# import libraries
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
from scipy.sparse import csr_matrix

#%%
# args
try: 
    argparser = ArgumentParser()
    argparser.add_argument('--items_folder', type=str, default='/mnt/E/Datasets/Ruten/item/activate_item')
    argparser.add_argument('--file_idx', type=int, default=0, help='file index of activate_item folder, use -1 to load all files at once')
    argparser.add_argument('--top_k', type=int, default=50)
    argparser.add_argument('--N_ROWS', type=int, default=None)

    args = argparser.parse_args()
    items_folder = args.items_folder
    file_idx = args.file_idx
    top_k = args.top_k

    N_ROWS = args.N_ROWS

except:
    items_folder = './results'
    file_idx = -1
    top_k = 50

    # test args
    N_ROWS = None

print(f'processing {file_idx}th file...')

#%%
# load item file from activate_item folder by file_idx
if file_idx >= 0:
    path_to_item_file = [file for file in os.listdir(items_folder) if file.endswith('.csv')][file_idx]
    items_df = pd.read_csv(os.path.join(items_folder, path_to_item_file), usecols=['product_name'])[:N_ROWS]
else:
    path_to_item_files = [file for file in os.listdir(items_folder) if file.endswith('.csv')]
    items_df = []
    for file in path_to_item_files:
        items_df.append(pd.read_csv(os.path.join(items_folder, file), usecols=['product_name']))
    items_df = pd.concat(items_df, ignore_index=True)[:N_ROWS]
    path_to_item_file = 'all'

# preprocess item_df
items_df['product_name'] = items_df['product_name'].map(html.unescape)
items_df['product_name'] = items_df['product_name'].fillna('')

#%%
# init tokenizer
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

#%%
# TF-IDF

# Paths to save/load the models
tfidf_model_path = 'tfidf_model.pkl'
tfidf_char_model_path = 'tfidf_char_model.pkl'
items_tfidf_matrix_path = 'items_tfidf_matrix.npz'
items_tfidf_matrix_char_path = 'items_tfidf_matrix_char.npz'

# Function to save the model and matrix
def save_model_and_matrix(model, matrix, model_path, matrix_path):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    np.savez(matrix_path, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=matrix.shape)

# Function to load the model and matrix
def load_model_and_matrix(model_path, matrix_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    loader = np.load(matrix_path)
    matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    return model, matrix

# Check if the models and matrices are already saved
if os.path.exists(tfidf_model_path) and os.path.exists(items_tfidf_matrix_path):
    tfidf, items_tfidf_matrix = load_model_and_matrix(tfidf_model_path, items_tfidf_matrix_path)
else:
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", tokenizer=tokenizer, ngram_range=(1,2))
    items_tfidf_matrix = tfidf.fit_transform(tqdm(items_df['product_name']))
    save_model_and_matrix(tfidf, items_tfidf_matrix, tfidf_model_path, items_tfidf_matrix_path)

if os.path.exists(tfidf_char_model_path) and os.path.exists(items_tfidf_matrix_char_path):
    tfidf_char, items_tfidf_matrix_char = load_model_and_matrix(tfidf_char_model_path, items_tfidf_matrix_char_path)
else:
    tfidf_char = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", analyzer='char')
    items_tfidf_matrix_char = tfidf_char.fit_transform(items_df['product_name'])
    save_model_and_matrix(tfidf_char, items_tfidf_matrix_char, tfidf_char_model_path, items_tfidf_matrix_char_path)

#%%
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

    return top_k_names

#%%