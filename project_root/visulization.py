from config import config
from 

# For each topic in the LDA model
for t in range(lda_gensim.num_topics):
    # Get the top 30 words for the topic
    plt.figure()
    plt.imshow(WordCloud(background_color='white').fit_words(dict(lda_gensim.show_topic(t, 30))))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()
