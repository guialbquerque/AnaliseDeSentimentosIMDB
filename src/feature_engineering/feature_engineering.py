import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk import tokenize
import pandas as pd
import seaborn as sns
nltk.download('stopwords')
from string import punctuation
from nltk.stem import PorterStemmer

def positive_words_cloud(text, column_text):

    positive = text.query("Sentiment == 1")
    all_words = ' '.join([text for text in positive[column_text]])
    model_cloud = WordCloud(width = 800, height = 500, max_font_size = 110, collocations = False)
    word_cloud = model_cloud.generate(all_words)
    fig = plt.figure(figsize = (12,8))
    plt.imshow(word_cloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.savefig(f'/home/guilherme/AIEnvironment/SentimentAnalysis/reports/figures/{column_text}_positive_words_cloud.png')
    plt.show()
    
def negative_words_cloud(text, column_text):

    positive = text.query("Sentiment == 0")
    all_words = ' '.join([text for text in positive[column_text]])
    model_cloud = WordCloud(width = 800, height = 500, max_font_size = 110, collocations = False)
    word_cloud = model_cloud.generate(all_words)
    fig = plt.figure(figsize = (12,8))
    plt.imshow(word_cloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.savefig(f'/home/guilherme/AIEnvironment/SentimentAnalysis/reports/figures/{column_text}_negative_words_cloud.png')
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
        
def remove_punctuation(df, df_column):
    punctuationn = list()
    for p in punctuation:
        punctuationn.append(p)
    stop_words = nltk.corpus.stopwords.words('english')
    total_to_remove = stop_words + punctuationn
    new_texts = list()
    for opinion in df[df_column]:
        news = list()
        word_texts = tokenize.WordPunctTokenizer().tokenize(opinion)
        for word in word_texts:
            if word not in total_to_remove:
                news.append(word)
        new_texts.append(' '.join(news))
    return new_texts

def normalize_corpus(df, df_column):
    punctuationn = list()
    for p in punctuation:
        punctuationn.append(p)
    stop_words = nltk.corpus.stopwords.words('english')
    total_to_remove = stop_words + punctuationn
    new_texts = list()
    for opinion in df[df_column]:
        opinion = opinion.lower()
        news = list()
        word_texts = tokenize.WordPunctTokenizer().tokenize(opinion)
        for word in word_texts:
            if word not in total_to_remove:
                news.append(word)
        new_texts.append(' '.join(news))
    return new_texts

def stemm_corpus(df, df_column): 
    ps = PorterStemmer()
    punctuationn = list()
    for p in punctuation:
        punctuationn.append(p)
    stop_words = nltk.corpus.stopwords.words('english')
    total_to_remove = stop_words + punctuationn
    new_texts = list()
    for opinion in df[df_column]:
        opinion = opinion.lower()
        news = list()
        word_texts = tokenize.WordPunctTokenizer().tokenize(opinion)
        for word in word_texts:
            if word not in total_to_remove:
                news.append(ps.stem(word))
        new_texts.append(' '.join(news))
    return new_texts
        
        