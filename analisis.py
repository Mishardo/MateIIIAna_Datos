  #2. Importacion de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
  
df = pd.read_csv('medicion.csv', encoding='cp1252', sep=',')
  
  
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
sns.kdeplot(df["indice_de_calidad_de_limpieza_adaptado"]) # Decidimos hacer el gráfico de densidad
plt.show()
#7 Detección y manejo de outliers
# Comprobamos si hay outliers de forma visual si es que hay puntos muy distantes de la mayoría
fig, ax = plt.subplots(figsize=(6,4))

ax.boxplot(df["indice_de_calidad_de_limpieza_adaptado"], vert=True)

plt.show()
# Ahora delimitamos usando el rango intercuartílico (IQR)
Q1 = df["indice_de_calidad_de_limpieza_adaptado"].quantile(0.25)
Q3 = df["indice_de_calidad_de_limpieza_adaptado"].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR 
outliers = df[(df["indice_de_calidad_de_limpieza_adaptado"] < limite_inferior) | (df["indice_de_calidad_de_limpieza_adaptado"] > limite_superior)]

print("outliers",outliers)
# 8. Análisis de correlación


# Analizamos la correlación visual entre todos los pares de columnas del df
sns.pairplot(df, hue='indice_de_calidad_de_limpieza_adaptado', diag_kind='kde', palette='viridis')
sns.pairplot(df, hue='medicion', diag_kind='kde', palette='viridis')
sns.pairplot(df, hue='periodo', diag_kind='kde', palette='viridis')
sns.pairplot(df, hue='estratificacion_geografica', diag_kind='kde', palette='viridis')
# Ahora las analizamos las correlaciones por pearson
print('Correlación Pearson: ', df['indice_de_calidad_de_limpieza_adaptado'].corr(df['medicion'], method='pearson')) # -0.20 correlación negativa debil, signfica que no hay una correlación fuerte entre ambas columnas
print('Correlación Spearman: ', df['indice_de_calidad_de_limpieza_adaptado'].corr(df['medicion'], method='spearman')) # -0.26 correlación negativa debil, signfica que no hay una correlación fuerte entre ambas columnas
print('Correlación Kendall: ', df['indice_de_calidad_de_limpieza_adaptado'].corr(df['medicion'], method='kendall')) # -0.19 correlación negativa debil, signfica que no hay una correlación fuerte entre ambas columnas

# Convertimos la variable categórica a variables dummy
df_corr = pd.get_dummies(
    df[["indice_de_calidad_de_limpieza_adaptado", "estratificacion_geografica"]],
    drop_first=True
)

plt.figure(figsize=(12,6))

sns.scatterplot(
    data=df,
    x='periodo',
    y='indice_de_calidad_de_limpieza_adaptado',
    hue='estratificacion_geografica',
    palette='viridis',
    s=100
)

plt.title("Índice de calidad de limpieza a través de los períodos")
plt.xlabel("Período")
plt.ylabel("Índice de calidad de limpieza")

plt.legend(title='Estratificación geográfica')

plt.show()

#9. Análisis Bivariado
var = 'indice_de_calidad_de_limpieza_adaptado'
data = pd.concat([df['medicion'], df['indice_de_calidad_de_limpieza_adaptado']], axis = 1)
data.plot.scatter(x = var, y = 'medicion', alpha = 0.5) 
# Se ve que no hay correlación entre ambas variables al estar los puntos tan dispersos en el gráfico, por lo que medición no es un predictor fuerte del indice de calidad de limpieza
#10 Transformaciones de datos
df_encoded = pd.get_dummies(df, columns=["estratificacion_geografica", "periodo"], drop_first=True) 
#11. Resumen y conclusiones

#12. Preparación para el modelado
# Variables independientes y dependiente
X = df_encoded.drop(["indice_de_calidad_de_limpieza_adaptado"], axis=1)
y = df_encoded["indice_de_calidad_de_limpieza_adaptado"]
# Separación correcta
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
modelo = LinearRegression() # usamos regresión lineal ya que buscamos predecir solo una columna, sino hubiese sido conveniente usar otro tipo de regresión, como regresión multiple
modelo.fit(X_train, y_train)
# Predicción
y_pred = modelo.predict(X_test)

# Evaluación
print(f"los resultados a predicir son: {y_test}, y la predicción es : {y_pred}" )
# Calculamos el	Error Cuadrático Medio(MSE) y su raiz cuadrada(RMSE) para ver que tan efectivo fue nuestro modelo en predecir los resultados del año 2024 en base a la zona.
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
# Al ser ambos tan cercanos a 0 implica un modelo excelente y que la predicción fue como se esperaba
