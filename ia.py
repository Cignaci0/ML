import tensorflow as tf
import numpy as np

# Datos originales
dias_restantes = np.arange(1, 365, dtype=float)
y_datos = dias_restantes

ph = np.linspace(8.5, 6.5, num=len(y_datos))
tds = np.linspace(500, 50, num=len(y_datos))

# Normalización: escala 0-1
ph_norm = (ph - 6.5) / (8.5 - 6.5)      # 6.5 -> 0, 8.5 -> 1
y_norm = (y_datos - 1) / (365 - 1)      # 5 -> 0, 365 -> 1

x_ph_norm = ph_norm.reshape(-1, 1)
y_norm = y_norm.reshape(-1, 1)

# Normalización: escala 0-1
tds_norm = (tds - 50) / (500 - 50)  
    # 6.5 -> 0, 8.5 -> 1
x_tds_norm = tds_norm.reshape(-1, 1)


# Modelo simple
modelo_ph = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

modelo_ph.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')

# Entrenamiento
print("Entrenando modelo pH...")
modelo_ph.fit(x_ph_norm, y_norm, epochs=1000, verbose=False)
modelo_ph.save("modelo_ph.keras")
print("Modelo entrenado!")


modelo_tds = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

modelo_tds.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')

# Entrenamiento
print("Entrenando modelo TDS...")
modelo_tds.fit(x_tds_norm, y_norm, epochs=1000, verbose=False)
modelo_tds.save("modelo_tds.keras")
print("Modelo entrenado!")

# Predicción para pH = 6.5
ph_nueva = np.array([[6.5]])
tds_nueva = np.array([[300]])
ph_nueva_norm = (ph_nueva - 6.5) / (8.5 - 6.5)
tds_nueva_norm = (tds_nueva - 50) / (500 - 50)

pred_norm_ph = modelo_ph.predict(ph_nueva_norm)
pred_norm_tds = modelo_tds.predict(tds_nueva_norm)


pred_dias_ph = pred_norm_ph* (365 - 1) + 1  # Desnormalizamos
pred_dias_tds = pred_norm_tds* (365 - 1) + 1  # Desnormalizamos


print(f"Días restantes de uso del filtro ph: {pred_dias_ph[0][0]:.2f}")
print(f"Días restantes de uso del filtro tds: {pred_dias_tds[0][0]:.2f}")

