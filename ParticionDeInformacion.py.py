import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
import matplotlib.pyplot as plt

# Cargar el dataset de datos
data = pd.read_csv('c:\\irisbin-irisbin.csv')
print(data.columns)

# Solicitar la cantidad de particiones y el porcentaje de entrenamiento y prueba
n_splits = int(input("Ingrese la cantidad de particiones (por ejemplo, 3 para validación cruzada): "))
test_size = float(input("Ingrese el porcentaje de patrones de prueba (por ejemplo, 0.3 para 30%): "))

# Técnica 1: Partición Aleatoria
X_train, X_test, y_train, y_test = train_test_split(data[['x1', 'x2', 'x3', 'x4']],
                                                    data[['y1', 'y2', 'y3']],
                                                    test_size=test_size, random_state=42)

# Técnica 2: Partición por Clase en 80/20
# Suponiendo que las etiquetas de clase son binarias
train_indices = []
test_indices = []
for class_label in ['y1', 'y2', 'y3']:
    class_data = data[data[class_label] == 1]
    train_idx, test_idx = train_test_split(range(len(class_data)), test_size=test_size, random_state=42)
    train_indices.extend(class_data.index[train_idx])
    test_indices.extend(class_data.index[test_idx])
X_train_class = data.iloc[train_indices][['x1', 'x2', 'x3', 'x4']]
y_train_class = data.iloc[train_indices][['y1', 'y2', 'y3']]
X_test_class = data.iloc[test_indices][['x1', 'x2', 'x3', 'x4']]
y_test_class = data.iloc[test_indices][['y1', 'y2', 'y3']]

# Técnica 3: Validación Cruzada
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(data)):
    X_train_cv = data.iloc[train_index][['x1', 'x2', 'x3', 'x4']]
    y_train_cv = data.iloc[train_index][['y1', 'y2', 'y3']]
    X_test_cv = data.iloc[test_index][['x1', 'x2', 'x3', 'x4']]
    y_test_cv = data.iloc[test_index][['y1', 'y2', 'y3']]


# Técnica 4: Partición Estratificada
splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
for train_index, test_index in splitter.split(data, data[['y1', 'y2', 'y3']]):
    X_train_strat = data.iloc[train_index][['x1', 'x2', 'x3', 'x4']]
    y_train_strat = data.iloc[train_index][['y1', 'y2', 'y3']]
    X_test_strat = data.iloc[test_index][['x1', 'x2', 'x3', 'x4']]
    y_test_strat = data.iloc[test_index][['y1', 'y2', 'y3']]

# Técnica 5: Partición Basada en Características
X_train_feat = data[['x1', 'x2']]
y_train_feat = data[['y1', 'y2', 'y3']]
X_test_feat = data[['x3', 'x4']]
y_test_feat = data[['y1', 'y2', 'y3']]

# Muestra una gráfica para cada técnica
for i, (X_train, X_test, y_train, y_test) in enumerate([(X_train, X_test, y_train, y_test),
                                                       (X_train_class, X_test_class, y_train_class, y_test_class),
                                                       (X_train_cv, X_test_cv, y_train_cv, y_test_cv),
                                                       (X_train_strat, X_test_strat, y_train_strat, y_test_strat),
                                                       (X_train_feat, X_test_feat, y_train_feat, y_test_feat)]):
    # Comprobar si 'x1' y 'x2' están presentes en X_train y 'x3' y 'x4' en X_test
    if 'x1' in X_train.columns and 'x2' in X_train.columns and 'x3' in X_test.columns and 'x4' in X_test.columns:
        plt.scatter(X_train['x1'], X_train['x2'], c='b', label='Entrenamiento')
        plt.scatter(X_test['x3'], X_test['x4'], c='r', label='Prueba')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.legend()
        plt.title(f'Técnica {i + 1}')
        plt.show()
