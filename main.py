from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()

# Load the trained preprocessing components
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load your trained model using joblib
with open("model_xgb.pkl", "rb") as f:
    model = joblib.load(f)

templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify/")
async def classify(request: Request, text: str = Form(...)):
    # Preprocess input text using TF-IDF vectorizer
    X_text = tfidf_vectorizer.transform([text])
    
    # Make prediction using the model
    prediction = model.predict(X_text)
    
    # Decode prediction using Label Encoder
    result = label_encoder.inverse_transform(prediction)[0]
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
