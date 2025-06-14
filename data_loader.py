import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dataset(folder_path):
    data = []
    with open(folder_path, 'r') as f:
        for line in f:
            data.append(line.strip())
    return data

def prepare_datasets(train_path, test_normal_path, test_abnormal_path):
    train_raw = load_dataset(train_path)
    test_normal_raw = load_dataset(test_normal_path)
    test_abnormal_raw = load_dataset(test_abnormal_path)

    vectorizer = TfidfVectorizer(token_pattern=r"\b\d+\b")

    x_train = vectorizer.fit_transform(train_raw).toarray().astype(np.float32)
    x_test_normal = vectorizer.transform(test_normal_raw).toarray().astype(np.float32)
    x_test_abnormal = vectorizer.transform(test_abnormal_raw).toarray().astype(np.float32)

    return x_train, x_test_normal, x_test_abnormal