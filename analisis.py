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