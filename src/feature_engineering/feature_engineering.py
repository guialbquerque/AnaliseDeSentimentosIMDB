import matplotlib.pyplot as plt
from wordcloud import WordCloud

def positive_words_cloud(text, column_text):

    positive = text.query("Sentiment == 1")
    all_words = ' '.join([text for text in positive[column_text]])
    model_cloud = WordCloud(width = 800, height = 500, max_font_size = 110, collocations = False)
    word_cloud = model_cloud.generate(all_words)
    fig = plt.figure(figsize = (12,8))
    plt.imshow(word_cloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.savefig('/home/guilherme/AIEnvironment/SentimentAnalysis/reports/figures/positive_words_cloud.png')
    plt.show()
    
def negative_words_cloud(text, column_text):

    positive = text.query("Sentiment == 0")
    all_words = ' '.join([text for text in positive[column_text]])
    model_cloud = WordCloud(width = 800, height = 500, max_font_size = 110, collocations = False)
    word_cloud = model_cloud.generate(all_words)
    fig = plt.figure(figsize = (12,8))
    plt.imshow(word_cloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.savefig('/home/guilherme/AIEnvironment/SentimentAnalysis/reports/figures/negative_words_cloud.png')
    plt.show()
    
    