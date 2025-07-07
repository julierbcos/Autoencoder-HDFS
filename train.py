import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from model import Autoencoder
from data_loader import prepare_datasets
from utils import evaluate

# Inicio del temporizador
start_time = time.time()

# Rutas de datos
train_file = '../output/hdfs/train'
test_normal_file = '../output/hdfs/test_normal'
test_abnormal_file = '../output/hdfs/test_abnormal'

# Carga de datos
x_train, x_test_normal, x_test_abnormal = prepare_datasets(train_file, test_normal_file, test_abnormal_file)

# Entrenamiento del modelo
model = Autoencoder(num_features=x_train.shape[1])
model.model.fit(x_train, x_train, batch_size=64, epochs=50, shuffle=True, verbose=1)

# Predicciones
pred_normal = model.model.predict(x_test_normal)
pred_abnormal = model.model.predict(x_test_abnormal)

# MSE por muestra
mse_normal = np.mean(np.square(x_test_normal - pred_normal), axis=1)
mse_abnormal = np.mean(np.square(x_test_abnormal - pred_abnormal), axis=1)

# Concatenación y etiquetas
y_true = np.array([0]*len(mse_normal) + [1]*len(mse_abnormal))
all_mse = np.concatenate((mse_normal, mse_abnormal))

# Thresholds del 95 al 99 percentil
percentiles = np.linspace(95, 99, num=10)
thresholds = np.percentile(mse_normal, percentiles)

precision_list = []
recall_list = []
f1_list = []

print("\n--- MÉTRICAS POR THRESHOLD ---")
for i, threshold in enumerate(thresholds):
    y_pred_tmp = (all_mse > threshold).astype(int)
    metrics_tmp = evaluate(y_true, y_pred_tmp)

    precision_list.append(metrics_tmp["Precision"])
    recall_list.append(metrics_tmp["Recall"])
    f1_list.append(metrics_tmp["F1"])

    print(f"\nThreshold: {int(percentiles[i])}")
    print(f"Threshold ({percentiles[i]:.1f}th percentile): {threshold:.6f}")
    print(f"Avg MSE - Normal: {np.mean(mse_normal):.7f}")
    print(f"Avg MSE - Abnormal: {np.mean(mse_abnormal):.7f}")
    print("Evaluación:", json.dumps(metrics_tmp, indent=2))

# Buscar el mejor threshold (máximo F1)
best_idx = np.argmax(f1_list)
best_f1 = f1_list[best_idx]
best_percentil = percentiles[best_idx]
best_threshold = thresholds[best_idx]

print("\n--- MEJOR THRESHOLD ---")
print(f"Mejor F1 Score: {best_f1:.4f}")
print(f"Percentil: {best_percentil:.2f}%")
print(f"Threshold real: {best_threshold:.6f}")

# Gráfico de métricas vs percentil del threshold
plt.figure(figsize=(10, 6))
plt.plot(percentiles, precision_list, label='Precisión', marker='o')
plt.plot(percentiles, recall_list, label='Recall', marker='s')
plt.plot(percentiles, f1_list, label='F1 Score', marker='^')
plt.axvline(x=best_percentil, color='gray', linestyle='--', label='Óptimo F1')

plt.xlabel("Umbral (Percentil del MSE normal)")
plt.ylabel("Valor de la métrica")
plt.title("Métricas vs Threshold (percentiles 95–99)")
plt.xticks(percentiles)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Crear carpeta de salida
os.makedirs('output', exist_ok=True)
plt.savefig("output/metricas_vs_threshold.png")
plt.close()
print("\nGráfico guardado en: output/metricas_vs_threshold.png")

# Añadir tiempo de ejecución
execution_time = time.time() - start_time
metrics_final["Execution_Time_Seconds"] = round(execution_time, 2)
