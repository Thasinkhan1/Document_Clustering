# inference.py
from config import config
from gensim import corpora
from gensim.models import LdaModel
from data_transformation import transforming_data
from gensim.corpora import Dictionary


def load_model_and_dict():
    model = LdaModel.load(config.SAVED_MODEL)
    dictionary = Dictionary.load(config.SAVED_DICT)

    return model, dictionary

def infer_topics(text):
    model, dictionary = load_model_and_dict()

    
    tokens = transforming_data.apply_tokenization([text])
    print("Tokenized:", tokens)

    tokens = tokens[0]
    lemmatized_tokens = transforming_data.lemmatized(tokens)
    print("Lemmatized:", lemmatized_tokens)

    bow_vector = dictionary.doc2bow(lemmatized_tokens)
    print("BoW:", bow_vector)

    topics = model.get_document_topics(bow_vector)
    topics = sorted(topics, key=lambda x: -x[1])
    for topic_id, topic in model.print_topics(num_words=10):
          print(f"Topic {topic_id}: {topic}")

    return topics


if __name__ == "__main__":
    test_text = "Crop leaves have yellow spots and fungal infection."
    topics = infer_topics(test_text)
    print("Predicted Topics (Topic ID, Score):")
    for topic_id, score in topics:
        print(f"Topic {topic_id}: {score:.4f}")
