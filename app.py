from flask import Flask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World Changes!</p>"

df = pd.read_csv("train.csv", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")
#df.info()
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)
#print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=text_clf.classes_)
disp.plot()
#plt.show()

# Making predictions and printing result
userInput = input("Insert your sentence here:\n")
prediction = text_clf.predict([userInput])
print(f"Prediction: {prediction[0]}")

quit()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
