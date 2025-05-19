from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo
modelo = joblib.load("modelo_lr.pkl")

def preprocesar_datos(datos):
    # Crear DataFrame inicial
    df = pd.DataFrame([datos])
    
    # Codificar variables categóricas
    df_encoded = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    
    # Asegurarse de que todas las columnas necesarias estén presentes
    columnas_requeridas = [
        'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'ChestPainType_ASY',
        'RestingECG_Normal', 'RestingECG_ST', 'RestingECG_LVH',
        'ExerciseAngina_N', 'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up', 'ST_Slope_Down'
    ]
    
    # Agregar columnas faltantes con valor 0
    for col in columnas_requeridas:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
    return df_encoded[columnas_requeridas]

@app.route('/', methods=['GET', 'POST'])
def formulario():
    prediccion = None
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            datos = {
                'Age': int(request.form['Age']),
                'Sex': request.form['Sex'],
                'ChestPainType': request.form['ChestPainType'],
                'RestingBP': int(request.form['RestingBP']),
                'Cholesterol': int(request.form['Cholesterol']),
                'FastingBS': int(request.form['FastingBS']),
                'RestingECG': request.form['RestingECG'],
                'MaxHR': int(request.form['MaxHR']),
                'ExerciseAngina': request.form['ExerciseAngina'],
                'Oldpeak': float(request.form['Oldpeak']),
                'ST_Slope': request.form['ST_Slope']
            }
            
            # Preprocesar datos
            input_data = preprocesar_datos(datos)
            
            # Hacer predicción
            pred = modelo.predict(input_data)[0]
            prediccion = "Presencia de enfermedad cardíaca" if pred == 1 else "Sin enfermedad cardíaca"
        except Exception as e:
            prediccion = f"Error en la predicción: {e}"

    return render_template('formulario.html', prediccion=prediccion)

if __name__ == '__main__':
    app.run()