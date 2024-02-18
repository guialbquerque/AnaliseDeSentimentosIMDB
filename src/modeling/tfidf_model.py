# All imports needed for this script
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def classification_text_tifdf(text, column_text, column_sentiment, model):
    tfidf = TfidfVectorizer(max_features = 1500, ngram_range = (1,2))
    tfidf_final = tfidf.fit_transform(text[column_text])
    train_X, test_X, train_y, test_y = train_test_split(tfidf_final, text[column_sentiment], random_state = 42)
    model_ml = model
    if isinstance(model_ml, (GaussianNB, MultinomialNB)):
        model_ml.fit(train_X.toarray(), train_y)
        Accuracy = model_ml.score(test_X.toarray(), test_y)
    else:
        model_ml.fit(train_X, train_y)
        Accuracy = model_ml.score(test_X, test_y)
    return Accuracy  