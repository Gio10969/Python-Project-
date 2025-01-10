#!/usr/bin/env python
# coding: utf-8

# Hola &#x1F600;
# 
# Soy **Hesus Garcia**  como "Jesús" pero con H. Sé que puede ser confuso al principio, pero una vez que lo recuerdes, ¡nunca lo olvidarás! &#x1F31D;	. Como revisor de código de Triple-Ten, estoy emocionado de examinar tus proyectos y ayudarte a mejorar tus habilidades en programación. si has cometido algún error, no te preocupes, pues ¡estoy aquí para ayudarte a corregirlo y hacer que tu código brille! &#x1F31F;. Si encuentro algún detalle en tu código, te lo señalaré para que lo corrijas, ya que mi objetivo es ayudarte a prepararte para un ambiente de trabajo real, donde el líder de tu equipo actuaría de la misma manera. Si no puedes solucionar el problema, te proporcionaré más información en la próxima oportunidad. Cuando encuentres un comentario,  **por favor, no los muevas, no los modifiques ni los borres**.
# 
# Revisaré cuidadosamente todas las implementaciones que has realizado para cumplir con los requisitos y te proporcionaré mis comentarios de la siguiente manera:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si todo está perfecto.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
# </div>
# 
# </br>
# 
# **¡Empecemos!**  &#x1F680;
# 
# 

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

# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a> Adelante Giovani, por favor procede a realizar tu código en este notebook. </div>
# 

# <div class="alert alert-block alert-info">
# Hola, muchas gracias por los comentarios del proyecto, tengo una consulta en el momento en el que empiece el codigo no tengo la dirección del archivo. este deveria pedirlo o usar una dirección ficticia de este mismo?.
# </div>
# 

# In[1]:


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

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Hola Giovani,
# 
# He notado que tienes varias dudas respecto al tratamiento de los datos y el enfoque del modelo. Como analista, es importante que tomes decisiones fundamentadas basadas en el entendimiento del problema y de los datos. A continuación, te brindaré algunas orientaciones para que puedas avanzar:
# 
# **Tratamiento de contratos sin fecha de finalización:**
# 
# - Los contratos que no tienen fecha de finalización representan clientes que aún están activos. Por lo tanto, puedes utilizar esta información para generar tu variable objetivo ('target').
# 
# - Te sugiero crear una nueva columna, por ejemplo, 'Churn', donde asignarás:
# 
#   - **1** si el cliente ha cancelado el contrato (es decir, si 'EndDate' tiene una fecha válida).
# 
#   - **0** si el cliente sigue activo (es decir, si 'EndDate' es 'No' o está vacío).
# 
# **Decisiones como analista:**
# 
# - Es esencial que definas claramente el objetivo de tu modelo. En este caso, queremos predecir la **probabilidad de cancelación** (churn) de los clientes. Por lo tanto, se trata de un problema de **clasificación binaria**.
# 
# - Aunque has calculado la duración del contrato, y esta puede ser una característica útil, el objetivo principal es predecir si un cliente cancelará o no. Puedes utilizar la duración como una de las variables predictoras, pero recuerda que el enfoque principal es la clasificación.
# 
# **Tratamiento de datos faltantes o incorrectos en 'TotalCharges':**
# 
# - Si encuentras valores nulos o incorrectos en 'TotalCharges', debes decidir si imputarlos o eliminarlos.
# 
# - Una buena práctica es analizar la cantidad y el porcentaje de datos faltantes. Si son pocos, podrías eliminarlos; si son significativos, considera imputarlos con métodos adecuados (media, mediana, etc.), siempre y cuando no introduzcan sesgos en el análisis.
# 
# **Selección de características importantes:**
# 
# - Para identificar las características más relevantes para tu modelo, puedes utilizar técnicas como **SHAP** o **Boruta**.
# 
#   - **SHAP (SHapley Additive exPlanations)** te ayuda a interpretar y visualizar el impacto de cada característica en las predicciones del modelo.
# 
#   - **Boruta** es un algoritmo de selección de características que te permite identificar las variables más importantes de manera objetiva.
# 
# **Siguientes pasos recomendados:**
# 
# - **Integración de datos:** Combina los diferentes conjuntos de datos ('contract', 'personal', 'internet', 'phone') utilizando 'customerID' como clave primaria. Esto te permitirá tener toda la información consolidada para cada cliente.
# 
# - **Generación de la variable objetivo:** Como mencioné, crea la variable 'Churn' basándote en 'EndDate'.
# 
# - **Análisis exploratorio de datos (EDA):** Realiza un EDA para entender mejor las relaciones entre las variables y cómo se correlacionan con la cancelación del servicio. Visualizaciones como histogramas, gráficos de barras y matrices de correlación pueden ser útiles.
# 
# - **Preprocesamiento de datos:**
# 
#   - **Codificación de variables categóricas:** Utiliza técnicas como One-Hot Encoding o Label Encoding para convertir variables categóricas en numéricas.
# 
#   - **Escalado de variables numéricas:** Si vas a utilizar algoritmos sensibles a la escala de las variables (como regresión logística o redes neuronales), considera normalizar o estandarizar tus variables numéricas.
# 
# - **Modelado:**
# 
#   - Dado que el objetivo es predecir la cancelación, enfócate en modelos de **clasificación**.
# 
#   - Puedes probar con modelos como regresión logística, árboles de decisión, Random Forest, XGBoost, entre otros.
# 
#   - Si deseas explorar la duración del contrato como una variable adicional, asegúrate de tratar correctamente los datos y justificar su inclusión en el modelo.
# 
# - **Evaluación del modelo:**
# 
#   - Utiliza métricas apropiadas para problemas de clasificación, especialmente en contextos de desbalance de clases. Las métricas clave incluyen **ROC-AUC**, **precisión**, **recall**, **F1-Score** y la matriz de confusión.
# 
#   - Aplica técnicas de validación cruzada para asegurar que tu modelo generaliza bien a datos no vistos.
# 
# **Recuerda:**
# 
# - Como analista, es fundamental que tomes decisiones basadas en el entendimiento del contexto y los datos. No temas explorar diferentes enfoques y justificar tus elecciones.
# 
# - Documenta cada paso y las razones detrás de tus decisiones. Esto es esencial para la transparencia y reproducibilidad del análisis.
# 
# ¡Ánimo, Giovani! Confío en que con estas orientaciones podrás avanzar en tu proyecto. Si tienes más preguntas o necesitas aclaraciones adicionales, no dudes en consultarme.
# 
# </div>
# 
