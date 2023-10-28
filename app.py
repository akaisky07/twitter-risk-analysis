from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('twitter_prediction.pkl')

# Load the CountVectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Home page
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    
    # Preprocess the tweet using the loaded vectorizer
    tweet_vectorized = vectorizer.transform([tweet]).astype(int)
    
    # Make the prediction using the pre-trained model
    prediction = model.predict(tweet_vectorized)
    a="yes"
    if prediction==0:
    	a="no"
    return render_template('index.html', prediction=a)

if __name__ == '__main__':
    app.run(debug=True)

