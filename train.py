import json
import numpy as np
from autoencoder_hdfs.model import Autoencoder
from autoencoder_hdfs.data_loader import prepare_datasets
from autoencoder_hdfs.utils import evaluate

train_file = 'data/processed/train'
test_normal_file = 'data/processed/test_normal'
test_abnormal_file = 'data/processed/test_abnormal'

x_train, x_test_normal, x_test_abnormal = prepare_datasets(train_file, test_normal_file, test_abnormal_file)

model = Autoencoder(num_features=x_train.shape[1])
model.model.fit(x_train, x_train, batch_size=32, epochs=20, shuffle=True)

# MSE para cada ejemplo
mse_normal = np.mean(np.power(x_test_normal - model.model.predict(x_test_normal), 2), axis=1)
mse_abnormal = np.mean(np.power(x_test_abnormal - model.model.predict(x_test_abnormal), 2), axis=1)

threshold = np.percentile(mse_normal, 95)
y_true = np.array([0]*len(mse_normal) + [1]*len(mse_abnormal))
y_pred = np.array([1 if x > threshold else 0 for x in np.concatenate((mse_normal, mse_abnormal))])

metrics = evaluate(y_true, y_pred)
with open('output/metrics_autoencoder.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(metrics)
