import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk import tokenize
import pandas as pd
import seaborn as sns
nltk.download('stopwords')

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
    

def plot_word_frequencies(text, column_text, quantity):
    
    all_words = ' '.join([text for text in text[column_text]])
    token = tokenize.WhitespaceTokenizer().tokenize(all_words)
    frequency = nltk.FreqDist(token)
    df_frequency = pd.DataFrame({"Palavra": list(frequency.keys()), "Frequência": list(frequency.values())})
    df_frequency = df_frequency.nlargest(n = quantity, columns = 'Frequência')
    plt.figure(figsize = (8,6))
    ax = sns.barplot(data = df_frequency, x = 'Palavra', y = 'Frequência')
    ax.set_xlabel('Palavra', fontsize=14)
    ax.set_ylabel('Frequência', fontsize=14)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis = 'x', colors = 'black')
    ax.tick_params(axis = 'y', colors = 'black')
    plt.show()


def remove_stop_words(df, df_column):
    stop_words = nltk.corpus.stopwords.words('english')
    new_texts = list()
    for opinion in df[df_column]:
        news = list()
        word_texts = tokenize.WhitespaceTokenizer().tokenize(opinion)
        for word in word_texts:
            if word not in stop_words:
                news.append(word)
        new_texts.append(' '.join(news))
    return new_texts
        
            