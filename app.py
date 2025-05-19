from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo, el scaler y la importancia de variables
modelo = joblib.load('modelo_lr.pkl')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')
feature_importance = joblib.load('feature_importance.joblib')

def preprocesar_datos(datos):
    # Crear DataFrame inicial
    df = pd.DataFrame([datos])
    
    # Convertir variables categóricas a dummy variables
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Asegurarse de que todas las columnas del modelo estén presentes
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Reordenar las columnas para que coincidan con el orden del modelo
    df = df[feature_names]
    
    # Escalar los datos
    df_scaled = scaler.transform(df)
    
    return df_scaled

def obtener_variables_importantes(datos_procesados, top_n=3):
    # Obtener los coeficientes del modelo para esta predicción
    coef = modelo.coef_[0]
    
    # Crear DataFrame con los coeficientes
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef
    })
    
    # Ordenar por valor absoluto de coeficiente
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    # Obtener las top N variables más importantes
    top_features = coef_df.head(top_n)
    
    # Formatear el resultado
    important_vars = []
    for _, row in top_features.iterrows():
        direction = "positiva" if row['coefficient'] > 0 else "negativa"
        important_vars.append({
            'name': row['feature'],
            'direction': direction,
            'importance': abs(row['coefficient'])
        })
    
    return important_vars

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener datos del formulario
        datos = {
            'Age': float(request.form['Age']),
            'Sex': request.form['Sex'],
            'ChestPainType': request.form['ChestPainType'],
            'RestingBP': float(request.form['RestingBP']),
            'Cholesterol': float(request.form['Cholesterol']),
            'FastingBS': float(request.form['FastingBS']),
            'RestingECG': request.form['RestingECG'],
            'MaxHR': float(request.form['MaxHR']),
            'ExerciseAngina': request.form['ExerciseAngina'],
            'Oldpeak': float(request.form['Oldpeak']),
            'ST_Slope': request.form['ST_Slope']
        }
        
        # Preprocesar datos
        datos_procesados = preprocesar_datos(datos)
        
        # Hacer predicción
        prediccion = modelo.predict(datos_procesados)[0]
        
        # Obtener variables importantes
        variables_importantes = obtener_variables_importantes(datos_procesados)
        
        return render_template('formulario.html', 
                             prediction=prediccion,
                             important_vars=variables_importantes)
    
    return render_template('formulario.html')

if __name__ == '__main__':
    app.run(debug=True)