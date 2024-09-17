import pandas as pd

def retrieve_cleaned_data() -> pd.DataFrame:
    """ 1. Retrieve data as pandas dataframe from Le Wagon public dataset.
        2. Drop rows containing missing values
        3. Drop duplicated values"""
    # Retrieve data
    data = pd.read_csv("https://wagon-public-datasets.s3.amazonaws.com/certification/da-ds-de/reviews.csv")
    print(f"✅ data has been retrieved -> Shape : {data.shape}")

    # Drop rows with missing values
    data = data.dropna()
    print(f"✅ Rows with missing values dropped -> Shape : {data.shape}")

    #Drop duplicated rows
    data = data.drop_duplicates()
    print(f"✅ Duplicated rows dropped -> Shape : {data.shape}")

    return data
