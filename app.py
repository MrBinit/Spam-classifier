from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained preprocessing components
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("Label_encoder.pkl")

# Load your trained model using joblib
with open("model_xgb.pkl", "rb") as f:
    model = joblib.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get input text from HTML form
    text = request.form['text']
    
    # Preprocess input text using TF-IDF vectorizer
    X_text = tfidf_vectorizer.transform([text])
    
    # Make prediction using the model
    prediction = model.predict(X_text)
    
    # Decode prediction using Label Encoder
    result = label_encoder.inverse_transform(prediction)[0]
    
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
