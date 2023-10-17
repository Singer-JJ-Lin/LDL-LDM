import os
import pickle

def save_state(file_path, scores):
    with open(file_path, "wb") as f:
        pickle.dump(scores, f)

def load_state(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError()
    
    with open(file_path, "rb") as f:
        return pickle.load(f)