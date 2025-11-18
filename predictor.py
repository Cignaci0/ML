from tensorflow.keras.models import load_model
import numpy as np
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


try:    
    modelo_ph = load_model("modelo_ph.keras")
    modelo_tds = load_model("modelo_tds.keras")
    logging.info("Modelos (pH y TDS) cargados correctamente.")
except Exception as e:
    logging.error(f"Error al cargar los modelos: {e}")
    modelo_ph = None
    modelo_tds = None

def predecir_vida_util(ph, tds):
    # Validar rangos
    if ph < 6.5 or ph > 8.5 or tds > 500 or tds < 50:
        logging.warn(f"Valores fuera de rango (pH: {ph}, TDS: {tds}). Devolviendo 0 días.")
        return 0.0  

    # Normalizar (Si los valores son válidos)
    ph_norm = (ph - 6.5) / (8.5 - 6.5)
    tds_norm = (tds - 50) / (500 - 50)

    ph_input = np.array([[ph_norm]])
    tds_input = np.array([[tds_norm]])

    # Predecir
    pred_norm_ph = modelo_ph.predict(ph_input, verbose=0)
    pred_norm_tds = modelo_tds.predict(tds_input, verbose=0)

    # Desnormalizar predicción
    pred_dias_ph = pred_norm_ph * (365 - 1) + 1
    pred_dias_tds = pred_norm_tds * (365 - 1) + 1

    # Tomar el valor más pequeño
    dias_restantes = min(pred_dias_ph[0][0], pred_dias_tds[0][0])
    
    # Devolver el valor
    return max(0, dias_restantes) 

@app.route("/predict", methods=["POST"])
def predic():

    if not modelo_ph or not modelo_tds:
        logging.error("Llamada a /predict pero los modelos no están cargados.")
        return jsonify({"error": "Los modelos no están cargados"}), 500
    
    try:
        data = request.json
        ph = float(data['ph'])
        tds = float(data['tds'])
        
        logging.info(f"Petición de predicción recibida: pH={ph}, TDS={tds}")
        
        dias_restantes = predecir_vida_util(ph, tds) 
        
        logging.info(f"Predicción: {dias_restantes:.2f} días")
        
        return jsonify({
            "dias_restantes": f"{dias_restantes:.2f}"
        })
    except Exception as e:
        logging.error(f"Error en la predicción: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)





