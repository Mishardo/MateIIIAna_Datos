import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

warnings.simplefilter("ignore")

df = pd.read_csv('medicion.csv', encoding='cp1252', sep=',')
print(df)
df["anio"] = df["periodo"].str.extract(r'(\d{4})').astype(int)
df_train = df[df["anio"].isin([2022, 2023])]
df_encoded = pd.get_dummies(df_train, columns=["estratificacion_geografica"])
X = df_encoded.drop(columns=[
    "indice_de_calidad_de_limpieza_adaptado",
    "periodo"
])

y = df_encoded["indice_de_calidad_de_limpieza_adaptado"]
df_2024 = df[df["anio"] == 2023].copy()

df_2024["anio"] = 2024  # simulamos el futuro

df_2024 = pd.get_dummies(df_2024, columns=["estratificacion_geografica"])

# Asegurar mismas columnas
df_2024 = df_2024.reindex(columns=X.columns, fill_value=0)
model = LinearRegression()

model.fit(X, y)
predicciones = model.predict(df_2024)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(mean_squared_error(y_test, y_pred))
df_2024 = df[df["anio"] == 2023].copy()

df_2024["anio"] = 2024  # simulamos el futuro

df_2024 = pd.get_dummies(df_2024, columns=["estratificacion_geografica"])

# Asegurar mismas columnas
df_2024 = df_2024.reindex(columns=X.columns, fill_value=0)

predicciones = model.predict(df_2024)