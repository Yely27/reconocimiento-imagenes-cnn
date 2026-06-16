import tensorflow as tf

# 1. Le decimos a Python que cargue tu modelo pesado
modelo_computadora = tf.keras.models.load_model("modelo_final_eurosat.keras")

# 2. Preparamos el "compresor" de modelos
convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo_computadora)

# 3. Le aplicamos una optimización para que sea ultra ligero en el celular
convertidor.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Hacemos la conversión mágica
modelo_celular = convertidor.convert()

# 5. Guardamos el nuevo archivo optimizado
with open("modelo_eurosat.tflite", "wb") as f:
    f.write(modelo_celular)

print("¡Listo! Tu modelo para celular se ha guardado como 'modelo_eurosat.tflite'")