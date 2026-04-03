import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = {
    "text":[
        "Government of India Railway recruitment apply now",
        "Ministry job application fee 5000 urgent hiring",
        "UPSC official notification civil service exam",
        "Pay money and get railway job immediately",
        "Indian army recruitment official portal",
        "Govt job guaranteed pay processing fee"
    ],
    "label":[0,1,0,1,0,1]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])

y = df["label"]

model = LogisticRegression()

model.fit(X,y)

pickle.dump(model,open("job_model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("Model trained")