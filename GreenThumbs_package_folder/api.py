# Basic libraries
import pandas as pd
import numpy as np
import json

from fastapi import FastAPI, UploadFile, File

# Import from .py files
from GreenThumbs_package_folder.api_functions.preprocessor_api import preprocess_features, get_tokenized,get_padded
from GreenThumbs_package_folder.api_functions.model_api import load_tokenizer,load_model,get_prediction

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'New project': 'This is the first app of my new project !'}

@app.post("/upload_and_predict_sentiment")
def create_upload_files(upload_file: UploadFile = File(...)):
    # Retrieve input data from json file
    json_data = json.load(upload_file.file)
    X_test = pd.DataFrame(json_data)

    # Preprocess data (i.e Removing whitespaces, Lowercasing,
                            # Removing numbers, Removing punctuation,
                            # Lemmatizing)
    X_test_preproc = preprocess_features(X_test['ReviewText'])

    # Tokenize
    tokenizer = load_tokenizer()
    X_test_tokens = get_tokenized(X_test_preproc, tokenizer)

    # Padding
    X_test_pad = get_padded(X_test_tokens, maxlen = 30)

    #  Load RNN model
    model = load_model('RNN')

    # Prediction
    prediction = get_prediction(X_test_pad, model)
    prediction = prediction.tolist()
    prediction_final = [np.round(pred[0],4) for pred in prediction]
    print(prediction_final)
    return {'Prediction' : prediction_final}
