# training_pipeline.py
from data_loading import data_loadings
from data_cleaning import cleaned_data
from data_transformation import transforming_data
from training import train
from gensim import corpora
from config import config
import json
import os

def pipeline():
    print("Loading raw data...")
    data = data_loadings.load_data()

    print("Cleaning text data...")
    cleaned_corpus = cleaned_data.clean_data(data)

    print("Tokenizing data...")
    tokens = transforming_data.apply_tokenization(cleaned_corpus)

    print("Applying lemmatization...")
    lemmatized_words = transforming_data.apply_lemmatizer(tokens)

    print("Saving lemmatized words...")
    with open(config.SAVE_LEMMAIZED_WORD_PATH, 'w') as f:
        json.dump(lemmatized_words, f)

    print("Creating dictionary and corpus...")
    dictionary = corpora.Dictionary(lemmatized_words)
    corpus = [dictionary.doc2bow(text) for text in lemmatized_words]

    print("Saving dictionary...")
    os.makedirs(config.SAVE_MODEL, exist_ok=True)
    dictionary.save("project_root/models/dictionary.dict")

    print("Training LDA model...")
    lda_model = train.train_lda_model()

    print("Saving trained model...")
    lda_model.save(config.SAVED_MODEL)

    print("✅ Training pipeline completed successfully.")

if __name__ == "__main__":
    pipeline()
