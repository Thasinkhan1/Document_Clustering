import pandas as pd
import json
import os
from config import config
from data_transformation import transforming_data
from data_cleaning import cleaned_data
from training import train
from gensim.models import LdaModel

def load_data():
    texts = []
    labels = []
    base_dir = [config.NEWS_PAPER_DATA_PATH,config.NEWS_MINI_DATA_PATH]
    for dirs in base_dir: 
       for root, dirs, files in os.walk(dirs):
           for file in files:
               file_path = os.path.join(root, file)
               try:
                   with open(file_path, 'r', encoding='latin-1') as f:
                       texts.append(f.read())
                       labels.append(os.path.basename(root))  # folder name = label
               except:
                   continue
    return pd.DataFrame({'text': texts, 'label': labels})



def save_lemmtized_word(lemmatized_word): 
    with open(config.SAVE_LEMMAIZED_WORD_PATH, 'w') as f:
        json.dump(lemmatized_word, f)
        
        
def loading_lemmatized_word():
    
    with open(config.SAVED_LEMMATIZED_WORD_PATH, 'r') as f:
        lemmatized_words = json.load(f)
    return lemmatized_words

def save_model():
    model = train.train_lda_model()
    model.save("project_root/models/lda_model.gensim")

def load_model():
    model = LdaModel.load(config.SAVED_MODEL)
    return model