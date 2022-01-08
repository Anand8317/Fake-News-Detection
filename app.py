from flask import Flask, render_template, request
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
import re





app = Flask(__name__)
loaded_model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open("cv.pickle", 'rb'))
ps = PorterStemmer()


def fake_news_det(news):
    

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
