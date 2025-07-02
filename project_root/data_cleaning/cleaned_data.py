from nltk.corpus import stopwords # type: ignore
import re
from config import config


def clean_data(data):
    corpus = []
    for i in range(len(data)):
       review = re.sub('[^a-zA-Z]',' ',data['text'][i])
       review = review.lower()
       review = review.split()
       review = [word for word in review if word not in set(stopwords.words('english'))]
       review = ' '.join(review)
       corpus.append(review)
       
    return corpus


