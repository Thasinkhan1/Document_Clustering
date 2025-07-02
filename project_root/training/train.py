from config import config
from gensim import corpora
from data_loading import data_loadings
import gensim.models.ldamodel

lemmatized_word = data_loadings.loading_lemmatized_word()
def train_lda_model():
    dictionary = corpora.Dictionary(lemmatized_word)
    corpus_gensim = [dictionary.doc2bow(text) for text in lemmatized_word]

    lda_gensim = gensim.models.ldamodel.LdaModel(
        corpus=corpus_gensim,
        id2word=dictionary,
        num_topics=10,
        passes=10,
        random_state=42
    )
    
    return lda_gensim