import json

def load(path):
    with open(path) as f:
        return json.load(f)

autoencoder = load("output/autoencoder_results.json")
logbert = load("output/logbert_results.json")

print("\n--- COMPARISON ---")
for metric in ["precision", "recall", "f1"]:
    print(f"{metric.upper():<10} | Autoencoder: {autoencoder[metric]:.4f} | LogBERT: {logbert[metric]:.4f}")
