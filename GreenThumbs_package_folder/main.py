# Basic libraries
import pandas as pd
import numpy as np
from colorama import Fore, Style

from sklearn.model_selection import train_test_split

# Import from .py files
from ml_logic.data import retrieve_cleaned_data
from ml_logic.preprocessor import get_features, get_target, preprocess_features

def main_program():
        # Retrieve the data and clean it
        reviews_df = retrieve_cleaned_data()

        # Get X and y from data
        X = get_features(reviews_df)
        y = get_target(reviews_df)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        print(f"✅ Train test split Done")

        # Preprocess X_train and X_test
        X_train_preproc = preprocess_features(X_train['ReviewText'])
        print(f"✅ X_train preprocessed")
        X_test_preproc = preprocess_features(X_test['ReviewText'])
        print(f"✅ X_test preprocessed")


        # Baseline calculation
        # Architecture
        # Training
        # Evaluation



        # Tokenize

        # Padding

        # RNN
        # Architecture
        # Training
        # Evaluation

        # CNN
        # Architecture
        # Training
        # Evaluation


       # Summary

def say_hello():
    print('Hello World !')

if __name__ == '__main__':
    try:
        main_program()

    except:
        import sys
        import traceback
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
