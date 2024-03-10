# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle

# # Load your trained model using pickle
# with open("model_xgb.pkl", "rb") as f:
#     model = pickle.load(f)

# # Define FastAPI app
# app = FastAPI()

# # Define request body model
# class TextData(BaseModel):
#     text: str

# # Define classification endpoint
# @app.post("/classify")
# async def classify(text_data: TextData):
#     text = text_data.text
#     # ML model to classify the text
#     prediction = model.predict([text])[0]
#     result = "Spam" if prediction == 1 else "Not Spam"
#     return {"result": result}


# if __name__ == "__main__":
#     uvicorn.run(app,host= '127.0.0.1', port = 5000)


import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # Using joblib for loading XGBoost model

# Load your trained model using joblib
with open("model_xgb.pkl", "rb") as f:
    model = joblib.load(f)

# Define FastAPI app
app = FastAPI()

# Define request body model
class TextData(BaseModel):
    text: str

# Define classification endpoint
@app.post("/classify")
async def classify(text_data: TextData):
    text = text_data.text
    # ML model to classify the text
    prediction = model.predict([text])[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return {"result": result}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
