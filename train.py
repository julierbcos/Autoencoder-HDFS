import os
import json
import numpy as np
from model import Autoencoder
from data_loader import prepare_datasets
from utils import evaluate

train_file = '../output/hdfs/train'
test_normal_file = '../output/hdfs/test_normal'
test_abnormal_file = '../output/hdfs/test_abnormal'

# Carga datasets
x_train, x_test_normal, x_test_abnormal = prepare_datasets(train_file, test_normal_file, test_abnormal_file)

# Modelo
model = Autoencoder(num_features=x_train.shape[1])
model.model.fit(x_train, x_train, batch_size=64, epochs=50, shuffle=True, verbose=1)

# Inferencia y MSE
pred_normal = model.model.predict(x_test_normal)
pred_abnormal = model.model.predict(x_test_abnormal)

mse_normal = np.mean(np.square(x_test_normal - pred_normal), axis=1)
mse_abnormal = np.mean(np.square(x_test_abnormal - pred_abnormal), axis=1)

# Ajuste de umbral
threshold = np.percentile(mse_normal, 95)

print("Threshold (95th percentile):", threshold)
print("Avg MSE - Normal:", np.mean(mse_normal))
print("Avg MSE - Abnormal:", np.mean(mse_abnormal))

# Clasificación
y_true = np.array([0]*len(mse_normal) + [1]*len(mse_abnormal))
y_pred = np.array([1 if x > threshold else 0 for x in np.concatenate((mse_normal, mse_abnormal))])

# Métricas
metrics = evaluate(y_true, y_pred)

# Guardar resultados
os.makedirs('output', exist_ok=True)
with open('output/metrics_autoencoder.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Evaluación:")
print(json.dumps(metrics, indent=2))