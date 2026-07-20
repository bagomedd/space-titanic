import kagglehub
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
def load_initial_df():
    train_path = os.path.join("", "../data/train.csv")
    path_to_data = "../data"
    if (not(os.path.exists(train_path))):
        print("Train data does not exist. Downloading")
        try:
            path_to_data = kagglehub.competition_download('spaceship-titanic',  output_dir="../data/")
        except:
            "You are not authorized. Please, read readme.md"
    sample_submission_df = pd.read_csv(path_to_data+"/sample_submission.csv")
    initial_df = pd.read_csv(path_to_data+"/train.csv")
    return initial_df