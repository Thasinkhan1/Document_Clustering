from config import config
from data_loading import data_loadings
from gensim.models import CoherenceModel
from gensim import corpora

lda_model = data_loadings.load_model()
lemmatized_word = data_loadings.loading_lemmatized_word()
dictionary = corpora.Dictionary(lemmatized_word)

def coherence():
    
    coherence_model = CoherenceModel(model=lda_model, texts=lemmatized_word, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    return coherence_score


print(f"Coherence Score is : {coherence():.4f}")
