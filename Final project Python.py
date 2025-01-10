
# ## PROYECTO 
# 
# 1) Preprocesamiento de Datos
# 
# - Técnicas de Filtrado: Aplicar filtrado univariado para identificar las características relevantes desde un punto de vista estadístico.
# 
# - Normalización y Escalado: Preprocesar los datos numéricos, incluyendo normalización y escalado, y realizar la codificación de variables categóricas utilizando One-Hot Encoding o Embedding, según corresponda.
# 
# 2) Selección de Características
# 
# - Aplicar métodos avanzados como Boruta para identificar las características más relevantes.
# 
# - Utilizar SHAP para interpretar y explicar las predicciones de los modelos, especialmente cuando se utilicen modelos complejos como árboles de decisión o redes neuronales.
# 
# 3) Entrenamiento de Modelos
# 
# - Entrenar varios modelos de clasificación y regresión (dependiendo del tipo de problema), como regresión logística, árboles de decisión, redes neuronales profundas, y modelos basados en ensambles (Random Forest, XGBoost).
# 
# - Evaluar diferentes combinaciones de hiperparámetros utilizando técnicas como la búsqueda en cuadrícula o búsqueda aleatoria.
# 
# 4) Evaluación de Modelos
# 
# - Utilizar precisión, recall, F1-Score y ROC-AUC para evaluar el rendimiento de los modelos de clasificación.
# 
# - Para tareas de regresión, evaluar con RMSE y MAE para medir la precisión de las predicciones numéricas.
# 
# 5) Optimización y Mejora Continua
# 
# - Interpretar los resultados con técnicas como SHAP para realizar ajustes en el modelo.
# 
# - Explorar mejoras a través de la ingeniería de características y el ajuste de hiperparámetros.
# 
# El plan ha sido mejorado para incluir técnicas avanzadas de selección de características (como Boruta y SHAP) y para garantizar que se utilicen las métricas adecuadas para problemas de clasificación desbalanceada, como F1-score y ROC-AUC. Además, el preprocesamiento de datos será más exhaustivo, aplicando técnicas de filtrado y embedding, lo cual mejora la calidad de los datos de entrada. Este enfoque optimizado asegurará que el modelo alcance el mejor rendimiento posible y sea interpretable.

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os

folder_path = '/datasets/final_provider'
files = os.listdir(folder_path)
files


# In[3]:


file_names = ['contract.csv', 'phone.csv', 'personal.csv', 'internet.csv']

dataframes = {}

for file in file_names:
    file_path = f"{folder_path}/{file}"
    dataframes[file] = pd.read_csv(file_path)


# In[4]:


dataframes['contract.csv']


# In[7]:


contract_df = dataframes['contract.csv']
print(contract_df.head())
print(contract_df.info())


# In[10]:


contract_df.describe()

print(contract_df.isnull().sum())


print(contract_df.dtypes)


# In[11]:


# Distribución de los Cargos Mensuales (MonthlyCharges)
plt.figure(figsize=(8, 6))
sns.histplot(contract_df['MonthlyCharges'], kde=True, bins=30)
plt.title('Distribución de Cargos Mensuales')
plt.xlabel('MonthlyCharges')
plt.show()

# Distribución de los Cargos Totales (TotalCharges)
plt.figure(figsize=(8, 6))
sns.histplot(contract_df['TotalCharges'], kde=True, bins=30)
plt.title('Distribución de Cargos Totales')
plt.xlabel('TotalCharges')
plt.show()


# In[12]:


# Convertir las fechas 'BeginDate' y 'EndDate' a formato datetime
contract_df['BeginDate'] = pd.to_datetime(contract_df['BeginDate'])
contract_df['EndDate'] = pd.to_datetime(contract_df['EndDate'], errors='coerce')

# Calcular la duración del contrato en días
contract_df['ContractDuration'] = (contract_df['EndDate'] - contract_df['BeginDate']).dt.days

# Descripción de la duración del contrato
print(contract_df['ContractDuration'].describe())

# Distribución de la duración del contrato
plt.figure(figsize=(8, 6))
sns.histplot(contract_df['ContractDuration'], kde=True, bins=30)
plt.title('Distribución de la Duración del Contrato (en días)')
plt.xlabel('Duración del Contrato (días)')
plt.show()


# Preguntas Aclaratorias
# ¿Cómo debemos tratar los contratos que no tienen fecha de finalización?:
# 
# ¿Se considera que estos clientes están activos, o debemos hacer alguna transformación adicional sobre ellos?
# Para el campo TotalCharges, en los casos donde los datos sean nulos o incorrectos, ¿deberíamos imputarlos (por ejemplo, con la mediana), o se espera que los eliminemos del análisis?.
# 
# ¿Existen características específicas que el equipo considere cruciales para el análisis y modelado (por ejemplo, PaymentMethod, Type, etc.)?
# 
# ¿Cuál es el objetivo principal del modelado?:
# 
# ¿Queremos predecir la duración del contrato, la probabilidad de cancelación, o existe otro objetivo específico?

# Hasta este punto, hemos realizado la limpieza de los datos y calculado la duración del contrato, lo que será una característica clave en los análisis posteriores. A medida que avancemos, podemos proceder con la ingeniería de características, explorar relaciones entre los datos y crear un modelo predictivo basado en las necesidades específicas del proyecto.

