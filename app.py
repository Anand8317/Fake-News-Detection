from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
cv = pickle.load(open("cv.pickle", 'rb'))


app = Flask(__name__)
#tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('train.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    #tfid_x_train = tfvect.fit_transform(x_train)
    #tfid_x_test = tfvect.transform(x_test)
    #input_data = [news]
    #vectorized_input_data = tfvect.transform(input_data)
    #prediction = loaded_model.predict(vectorized_input_data)

    review = re.sub('[^a-zA-Z]', ' ', news)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    news = [review]

    news = cv.transform(news).toarray()

    ans = loaded_model.predict(news)
    return int(ans)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)