import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv('medicion.csv', encoding='cp1252', sep=',')
df.info()
df.describe()
df.isnull().sum()
sns.histplot(df["indice_de_calidad_de_limpieza_adaptado"], kde=True)
plt.show()
sns.lineplot(x="periodo", y="indice_de_calidad_de_limpieza_adaptado", data=df)
plt.show()
df_encoded = pd.get_dummies(df, columns=["estratificacion_geografica"], drop_first=True)
train = df_encoded[df_encoded["periodo"].isin([2022, 2023])]
test = df_encoded[df_encoded["periodo"] == 2024]
X_train = train.drop(["indice_de_calidad_de_limpieza_adaptado"], axis=1)
y_train = train["indice_de_calidad_de_limpieza_adaptado"]

X_test = test.drop(["indice_de_calidad_de_limpieza_adaptado"], axis=1)
y_test = test["indice_de_calidad_de_limpieza_adaptado"]
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

plt.scatter(y_test, y_pred)
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Real vs Predicho")
plt.show()
predicciones_2024 = pd.DataFrame({
    "Real": y_test,
    "Predicho": y_pred
})

print(predicciones_2024.head())