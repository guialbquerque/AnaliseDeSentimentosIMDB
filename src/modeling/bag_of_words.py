# All imports needed for this script
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud



def classification_text_bow(df, column_text, column_sentiment, model):

    vector = CountVectorizer(max_features = 100, lowercase = True)
    BagOfWords = vector.fit_transform(df[column_text])
    train_X, test_X, train_y, test_y = train_test_split(BagOfWords,
                                                        df[column_sentiment],
                                                        test_size = 0.33,
                                                        random_state = 42)
    model = model
    if model == GaussianNB() or model == MultinomialNB():
      model.fit(train_X.todense(), train_y)
      Accuracy = model.score(test_X.todense(), test_y)
    else:
      model.fit(train_X, train_y)
      Accuracy = model.score(test_X, test_y)
    return Accuracy