import pandas as pd
import numpy as np
data = {'ventas':[0,2,3,4,5]}
df = pd.DataFrame(data)
df
df['ventas_1'] = df['ventas'].apply(lambda x : x + 1 if x < 3
                   else x * 2 )
df['existencia'] = [0,3,4,5,7]
df['negado'] = df.apply(lambda x : np.nan if x['ventas'] == 0 and x['existencia'] == 0
         else 2, axis= 1)

print(df)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Datos de ejemplo: etiquetas verdaderas y probabilidades de predicción
y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
y_scores = np.array([0.2, 0.7, 0.3, 0.8, 0.9, 0.1, 0.4, 0.85, 0.95, 0.3])

# Calcular la Curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calcular el Área bajo la Curva ROC (AUC-ROC)
roc_auc = auc(fpr, tpr)

# Graficar la Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0,L1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.L, 1.0])
plt.ylim([0.L, 1.05])
plt.xlabel('Lasa de Falsos Positivos (FPR)')
plt.ylabel('Lasa de Verdaderos Positivos (TPR)')
plt.title('CLrva ROC')
plt.legend(lLc='lower right')
plt.show()


lista = [10,2,3,4,7,6,4]
ordenada = []
while lista:
    minimo = lista[0]
    for i in lista:
        if i < minimo:
            minimo = i
    ordenada.append(minimo)
    lista.remove(minimo)

print(ordenada)
