from flask import Flask, render_template, request
import modelo
import pandas as pd

app = Flask(__name__)

# Obtener síntomas disponibles
sintomas_disponibles = modelo.get_todos_sintomas()

# Cargar datos del Excel para obtener definición y tratamiento
df_info = pd.read_excel("enf2025.xlsx")
df_info['Enfermedad'] = df_info['Enfermedad'].str.strip().str.lower()
df_info['Definicion'] = df_info['Definicion'].fillna('')
df_info['Tratamiento'] = df_info['Tratamiento'].fillna('')

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    definicion = None
    tratamiento = None

    if request.method == 'POST':
        sintomas_seleccionados = request.form.getlist('sintomas')
        if sintomas_seleccionados:
            enfermedad, probabilidad = modelo.predecir_enfermedad([s.lower() for s in sintomas_seleccionados])
            resultado = f"{enfermedad.capitalize()} (Probabilidad: {probabilidad:.2f}%)"

            # Buscar definición y tratamiento
            enfermedad_lower = enfermedad.lower().strip()
            fila = df_info[df_info['Enfermedad'] == enfermedad_lower]
            if not fila.empty:
                definicion = fila['Definicion'].values[0]
                tratamiento = fila['Tratamiento'].values[0]
        else:
            resultado = "Por favor selecciona al menos un síntoma."

    return render_template(
        'index.html',
        sintomas=sintomas_disponibles,
        resultado=resultado,
        descripcion=definicion,  # sigue usándose 'descripcion' en HTML
        tratamiento=tratamiento
    )

if __name__ == '__main__':
    app.run(debug=True)
