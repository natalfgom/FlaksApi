from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar modelo
modelo = joblib.load("modelo_lr.pkl")

# Cargar codificadores si los tienes (opcional)

@app.route('/', methods=['GET', 'POST'])
def formulario():
    prediccion = None
    if request.method == 'POST':
        try:
            Age = int(request.form['Age'])
            Sex = request.form['Sex']
            ChestPainType = request.form['ChestPainType']
            RestingBP = int(request.form['RestingBP'])
            Cholesterol = int(request.form['Cholesterol'])
            FastingBS = int(request.form['FastingBS'])
            RestingECG = request.form['RestingECG']
            MaxHR = int(request.form['MaxHR'])
            ExerciseAngina = request.form['ExerciseAngina']
            Oldpeak = float(request.form['Oldpeak'])
            ST_Slope = request.form['ST_Slope']

            input_data = pd.DataFrame([{
                'Age': Age,
                'Sex': Sex,
                'ChestPainType': ChestPainType,
                'RestingBP': RestingBP,
                'Cholesterol': Cholesterol,
                'FastingBS': FastingBS,
                'RestingECG': RestingECG,
                'MaxHR': MaxHR,
                'ExerciseAngina': ExerciseAngina,
                'Oldpeak': Oldpeak,
                'ST_Slope': ST_Slope
            }])

            # IMPORTANTE: aplicar misma transformación que en entrenamiento aquí
            # input_data = preprocesar(input_data) si tienes un pipeline

            pred = modelo.predict(input_data)[0]
            prediccion = "Presencia de enfermedad cardíaca" if pred == 1 else "Sin enfermedad cardíaca"
        except Exception as e:
            prediccion = f"Error en la predicción: {e}"

    return render_template('formulario.html', prediccion=prediccion)

if __name__ == '__main__':
    app.run(debug=True)