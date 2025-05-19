from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo
modelo = joblib.load("modelo_lr.pkl")

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
            
            # Crear DataFrame y hacer predicción
            input_data = pd.DataFrame([datos])
            pred = modelo.predict(input_data)[0]
            prediccion = "Presencia de enfermedad cardíaca" if pred == 1 else "Sin enfermedad cardíaca"
        except Exception as e:
            prediccion = f"Error en la predicción: {e}"

    return render_template('formulario.html', prediccion=prediccion)

if __name__ == '__main__':
    app.run()