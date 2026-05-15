  #2. Importacion de datos
  import pandas as pd
  import matplotlib.pyplot as plt
  import numpy as np
  import seaborn as sns
  import warnings
  
  warnings.simplefilter("ignore")
  
  df = pd.read_csv('data_nan.csv', encoding='cp1252', sep=',')
  
  
  #3. Revision de los datos
  df.head()
  df.shape #(filas, columnas)
  df.info()
  df.dtypes
  
  # 4 Revision de datos duplicados/faltantes
  df.isnull().sum()
  df.duplicated()
#5 Análisis estadístico descriptivo, utilizar .describe() para un primer análisis de las variables numéricas. Para variables categóricas usar .unique() o .value_counts() para ver las distribuciones de categorías. Calcular medidas como sesgo y curtosis.
# Utilizamos solo la que tiene el indice de limpieza ya que de las demás no tiene efecto usar el sesgo o curtosis
print(df["indice_de_calidad_de_limpieza_adaptado"].describe())
print(df.value_counts()) # Filtró las columnas que tenían valores repetidos
print(df["indice_de_calidad_de_limpieza_adaptado"].kurt()) # -0.62, significa desviación un poco más plana que una normal
print(df["indice_de_calidad_de_limpieza_adaptado"].skew()) # 0.47, significa que es skewness positivo
#6 Visualización de la distribución de datos, utilizar histogramas o gráficos de densidad para ver la distribución de las variables numéricas. Utilizar gráficos de barra para ver la distribución de las variables categóricas.
