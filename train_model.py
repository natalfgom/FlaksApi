import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Cargar el dataset
print("Cargando dataset...")
df = pd.read_csv('heart.csv')

# Mostrar información básica
print("\nForma del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())

# Separar features y target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Convertir variables categóricas a numéricas
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
X = pd.get_dummies(X, columns=categorical_columns)

# Guardar los nombres de las columnas para uso posterior
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.joblib')

# Dividir los datos
print("\nDividiendo datos en train y test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las features
print("Escalando features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
print("\nEntrenando modelo de regresión logística...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Calcular importancia de variables
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Hacer predicciones
y_pred = model.predict(X_test_scaled)

# Mostrar rendimiento del modelo
print("\nRendimiento del modelo:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar el modelo, el scaler y la importancia de variables
print("\nGuardando modelo, scaler e importancia de variables...")
joblib.dump(model, 'modelo_lr.pkl')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(feature_importance, 'feature_importance.joblib')

print("\n¡Entrenamiento completado con éxito!")
print("Archivos guardados:")
print("- modelo_lr.pkl")
print("- scaler.joblib")
print("- feature_names.joblib")
print("- feature_importance.joblib") 