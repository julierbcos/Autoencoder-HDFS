from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(y_true, y_pred):
    return {
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }