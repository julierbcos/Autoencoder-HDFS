import os
import json
import numpy as np
import matplotlib.pyplot as plt
from model import Autoencoder
from data_loader import prepare_datasets
from utils import evaluate

# Rutas
train_file = '../output/hdfs/train'
test_normal_file = '../output/hdfs/test_normal'
test_abnormal_file = '../output/hdfs/test_abnormal'

# Carga datasets
x_train, x_test_normal, x_test_abnormal = prepare_datasets(train_file, test_normal_file, test_abnormal_file)

# Entrenamiento del modelo
model = Autoencoder(num_features=x_train.shape[1])
model.model.fit(x_train, x_train, batch_size=64, epochs=50, shuffle=True, verbose=1)

# Inferencia
pred_normal = model.model.predict(x_test_normal)
pred_abnormal = model.model.predict(x_test_abnormal)

# Errores de reconstrucción (MSE por muestra)
mse_normal = np.mean(np.square(x_test_normal - pred_normal), axis=1)
mse_abnormal = np.mean(np.square(x_test_abnormal - pred_abnormal), axis=1)

# Etiquetas reales
y_true = np.array([0]*len(mse_normal) + [1]*len(mse_abnormal))
all_mse = np.concatenate((mse_normal, mse_abnormal))

# Thresholds del percentil 95 al 98
percentiles = np.linspace(95, 98, num=10)
thresholds = np.percentile(mse_normal, percentiles)

precision_list = []
recall_list = []
f1_list = []
threshold_vals = []

for threshold in thresholds:
    y_pred_tmp = np.array([1 if x > threshold else 0 for x in all_mse])
    metrics_tmp = evaluate(y_true, y_pred_tmp)

    precision_list.append(metrics_tmp["Precision"])
    recall_list.append(metrics_tmp["Recall"])
    f1_list.append(metrics_tmp["F1"])
    threshold_vals.append(threshold)

# Gráfico de métricas vs threshold
plt.figure(figsize=(10, 6))
plt.plot(threshold_vals, precision_list, label='Precisión', marker='o')
plt.plot(threshold_vals, recall_list, label='Recall', marker='s')
plt.plot(threshold_vals, f1_list, label='F1 Score', marker='^')
plt.xlabel("Threshold (valor real)")
plt.ylabel("Valor de la métrica")
plt.title("Métricas vs Threshold (percentiles 95–98)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar gráfico
os.makedirs('output', exist_ok=True)
plt.savefig("output/metricas_vs_threshold.png")
plt.close()
print("Gráfico guardado en: output/metricas_vs_threshold.png")

# Evaluación final con threshold al percentil 98
threshold = np.percentile(mse_normal, 98)
print("Threshold (98th percentile):", threshold)
print("Avg MSE - Normal:", np.mean(mse_normal))
print("Avg MSE - Abnormal:", np.mean(mse_abnormal))

# Clasificación con threshold final
y_pred = np.array([1 if x > threshold else 0 for x in all_mse])

# Métricas finales
metrics = evaluate(y_true, y_pred)

# Guardar resultados
with open('output/metrics_autoencoder.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Evaluación final (percentil 98):")
print(json.dumps(metrics, indent=2))