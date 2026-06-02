import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = pd.read_csv("train_fruit.csv")

texts = data["text"]
labels = data["label"]

#Convert text into numbers:
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

#train ai model
model = MultinomialNB()
model.fit(X, labels)

#save thi model
joblib.dump(model, "fruit_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully!")