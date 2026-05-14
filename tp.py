import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Carga de datos

df = pd.read_csv("medicion.csv")

# Codificación de variables categóricas (AHORA INCLUYE periodo)
df_encoded = pd.get_dummies(df, columns=["estratificacion_geografica", "periodo"], drop_first=True)
# Variables independientes y dependiente
X = df_encoded.drop(["indice_de_calidad_de_limpieza_adaptado"], axis=1)
y = df_encoded["indice_de_calidad_de_limpieza_adaptado"]
# Separación correcta
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)
# Predicción
y_pred = modelo.predict(X_test)

# Evaluación
print(f"los resultados a predicir son: {y_test}, y la predicción es : {y_pred}" )
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)