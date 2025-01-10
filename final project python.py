#!/usr/bin/env python
# coding: utf-8

# # Hola &#x1F600;
# 
# Soy **Hesus Garcia**, revisor de código de Triple Ten, y voy a examinar el proyecto que has desarrollado recientemente. Si encuentro algún error, te lo señalaré para que lo corrijas, ya que mi objetivo es ayudarte a prepararte para un ambiente de trabajo real, donde el líder de tu equipo actuaría de la misma manera. Si no puedes solucionar el problema, te proporcionaré más información en la próxima oportunidad. Cuando encuentres un comentario,  **por favor, no los muevas, no los modifiques ni los borres**. 
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

# # Plan de Trabajo para Proyecto en Python
# 
# 1. Análisis Exploratorio de Datos (EDA)
# El primer paso del proyecto será realizar un análisis exploratorio de datos (EDA) para entender mejor el conjunto de datos y su estructura. Esto implica lo siguiente:
# 
# Carga del Conjunto de Datos: Importar los datos y verificar su estructura (número de filas y columnas, tipos de datos).
# Describir el Conjunto de Datos: Analizar estadísticas básicas de las variables, como medias, medianas, valores nulos, y distribuciones.
# 
# Identificación de Valores Faltantes: Revisar la cantidad de valores faltantes por columna y si es necesario tratarlos.
# Detección de Outliers: Visualizar las distribuciones de las variables numéricas para detectar posibles valores atípicos.
# Visualización de Datos: Crear gráficos de barras, histogramas, y gráficos de dispersión para identificar patrones y relaciones entre las variables.
# 
# 2. Lista de Preguntas Aclaratorias
# Al finalizar el análisis exploratorio de datos, podrían surgir algunas preguntas para el líder del equipo, tales como:
# 
# ¿Cómo debemos manejar los valores faltantes?: En caso de haber muchas celdas con datos faltantes, ¿debemos eliminar esas filas/columnas o aplicar alguna técnica de imputación?
# 
# ¿Qué variables son más relevantes para el problema?: ¿Existe alguna variable en el conjunto de datos que debamos priorizar o que tenga una mayor relevancia según los conocimientos del dominio?
# 
# ¿Hay algún tipo de preprocesamiento específico esperado?: Dependiendo del dominio, ¿deberíamos realizar transformaciones adicionales, como normalización, escalado, o codificación de variables categóricas?
# 
# ¿Qué métricas se utilizarán para evaluar el modelo?: Es importante saber si nos vamos a centrar en una métrica específica como precisión, F1, RMSE, o alguna otra.
# 
# ¿Existen restricciones de tiempo de ejecución o recursos?: ¿Hay alguna restricción de tiempo o recursos que debamos considerar al entrenar el modelo, como el uso de GPU o CPU?
# 
# 3. Plan Aproximado para Resolver la Tarea
# Paso 1: Preprocesamiento de los Datos
# 
# Objetivo: Limpiar y preparar los datos para que sean aptos para el modelado.
# Acciones: Tratar los valores faltantes, codificar las variables categóricas, normalizar o estandarizar los valores numéricos, y dividir los datos en conjuntos de entrenamiento y prueba.

# Paso 2: Selección de Características
# 
# Objetivo: Reducir la dimensionalidad y mejorar la eficiencia del modelo.
# 
# Acciones: Evaluar las correlaciones entre las variables para identificar cuáles son más relevantes. Podríamos aplicar técnicas como selección automática o eliminación de características menos útiles.
# 
# Paso 3: Entrenamiento de Modelos
# 
# Objetivo: Probar diferentes modelos y optimizarlos para lograr el mejor rendimiento posible.
# 
# Acciones: Entrenar varios modelos, como regresión logística, árboles de decisión, y modelos basados en ensambles. Evaluar su rendimiento utilizando validación cruzada y ajustar hiperparámetros.
# 
# Paso 4: Evaluación del Modelo
# 
# Objetivo: Medir el rendimiento de los modelos entrenados en el conjunto de prueba.
# 
# Acciones: Evaluar las métricas clave como la precisión, recall, F1, o RMSE. Comparar el rendimiento de los modelos y elegir el más adecuado para el problema.
# 
# Paso 5: Conclusiones y Mejora Continua
# 
# Objetivo: Interpretar los resultados y decidir sobre posibles mejoras.
# 
# Acciones: Basado en los resultados del modelo, explorar opciones para mejorarlo, como el ajuste de hiperparámetros, ingeniería de características, o el uso de modelos más avanzados.

# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# 
# **Aspectos Positivos:**
# - El plan de trabajo está bien estructurado y cubre los pasos clave para desarrollar un proyecto de ciencia de datos, lo cual es excelente. La metodología de trabajo sigue un flujo lógico desde la exploración de datos hasta la evaluación del modelo, lo que asegura una progresión clara en el desarrollo del proyecto.
# - La inclusión de una sección de preguntas aclaratorias es muy útil, ya que esto permitirá ajustar el enfoque del proyecto según los requisitos del dominio y los lineamientos del equipo.
# - Se han considerado varios modelos y métodos de evaluación, lo cual es esencial para encontrar el mejor enfoque para el problema en cuestión.
# 
# **Aspectos a Mejorar:**
# 
# 1. **Selección de Características**:
#    - Aunque se menciona la selección de características, es importante profundizar en técnicas más avanzadas. Además de evaluar correlaciones, puedes utilizar técnicas como **Boruta** y **SHAP**. 
#      - **Boruta** es un método de selección de características basado en árboles de decisión que te ayudará a identificar las variables más relevantes.
#      - **SHAP** (SHapley Additive exPlanations) te permite interpretar la importancia de las características en modelos complejos.
#    - Aparte del análisis de correlación, te sugiero evitar incluir variables categóricas o booleanas en matrices de correlación, ya que pueden dar resultados incorrectos.
# 
# 2. **Preprocesamiento y Filtrado**:
#    - En la sección de preprocesamiento, puedes ampliar la explicación de las técnicas de **filtrado** y **embedding**.
#      - **Filtrado**: Basado en la relevancia estadística de las características sin tener en cuenta el modelo, por ejemplo, utilizando métodos como análisis univariado o test estadísticos.
#      - **Embedding**: Se refiere a la selección de características integrada en el modelo, donde el propio modelo selecciona las características más importantes durante su entrenamiento (e.g., en redes neuronales).
# 
# 3. **Métricas de Evaluación**:
#    - Además de las métricas mencionadas (precisión, recall, RMSE), asegúrate de incluir también el **F1-score** cuando trabajes con datos desbalanceados, ya que esta métrica te permitirá equilibrar entre la precisión y el recall.
#    - Para problemas de clasificación, puedes considerar el uso de la **curva ROC-AUC**, que mide la capacidad del modelo para distinguir entre las clases.
# 
# </div>
# 
