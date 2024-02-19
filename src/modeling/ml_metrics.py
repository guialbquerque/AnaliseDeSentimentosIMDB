# All imports needed for this script
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def get_roc_curve_pr(df, column_text, column_sentiment, model):
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
    tfidf_final = tfidf.fit_transform(df[column_text])
    train_X, test_X, train_y, test_y = train_test_split(tfidf_final, df[column_sentiment], random_state=42)
    ml_model = model
    if isinstance(ml_model, (GaussianNB, MultinomialNB)):
        ml_model.fit(train_X.toarray(), train_y)
        model_predict = ml_model.predict(test_X.toarray())
        model_predict_prob = ml_model.predict_proba(test_X.toarray())
    else:
        ml_model.fit(train_X, train_y)
        model_predict = ml_model.predict(test_X)
        model_predict_prob = ml_model.predict_proba(test_X)
    return test_y, model_predict, model_predict_prob, ml_model