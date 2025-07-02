from config import config
import gensim # type: ignore
import spacy # type: ignore
import json
from data_cleaning import cleaned_data
from data_loading import data_loadings


def tokenize(text):
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    return tokens
def apply_tokenization(corpus):
    tokenized_text = []
    
    for doc in corpus:
        tokenized_text.append(tokenize(doc))
        
    return tokenized_text

nlp = spacy.load('en_core_web_sm',disable=['parser','ner'])
def lemmatized(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]



