# Detector de crisis epilépticas en EEG continuo

Material reproducible del experimento **TCN–GRU de dos escalas temporales (2 y 4 segundos)**.
Resultado histórico de prueba cruzada: **114/168 crisis, sensibilidad 67.86%, 1.216 falsas
alarmas por hora sin crisis y precisión por evento 10.85%**. La demora mediana entre
las detectadas fue 11 s. El resultado es exploratorio y no acredita utilidad clínica.

[Abrir el cuaderno en Colab](https://colab.research.google.com/github/Armandogma/detector-eeg/blob/main/Colab_Detector_EEG.ipynb)

## Qué contiene

- `Colab_Detector_EEG.ipynb` descarga y verifica el código, copia los EDF necesarios desde Drive y ejecuta las etapas en orden.
- `codigo_congelado.zip` contiene las 22 entradas originales del paquete científico y sus huellas. Incluye configuración, inventario, anotaciones y particiones. El cuaderno verifica su SHA-256 antes de extraerlo.
- `fuentes/` permite leer esas mismas fuentes directamente en GitHub. Sus archivos son idénticos a los del ZIP. El cuaderno ejecuta el paquete comprobado, no una variante editada.
- `METODO.md` explica datos, entrenamiento, elección de época y evaluación.
- `resultados/` conserva recuentos históricos por ronda, crisis y duración. No contiene señales EEG.
- `verificar_entrega.py` comprueba las huellas, las cinco divisiones y los recuentos sin entrenar.

## Cómo ejecutarlo

1. Obtén CHB-MIT versión 1.0.0 desde [PhysioNet](https://physionet.org/content/chbmit/1.0.0/), DOI [10.13026/C2K01R](https://doi.org/10.13026/C2K01R). Respeta su licencia Open Data Commons Attribution 1.0.
2. Coloca las carpetas `chb01`, `chb02`, etc., con sus EDF, resúmenes y anotaciones en `Mi unidad/base de datos` de Google Drive. No basta con un CSV de etiquetas.
3. Abre el cuaderno y selecciona una GPU. La referencia histórica usó Tesla T4 y PyTorch 2.11.0. El cuaderno registra las versiones reales. Una sesión nueva puede usar otras versiones y no garantiza identidad numérica.
4. Ejecuta las celdas en orden. El SSD `/content/cache_pediatrica_ssd` es temporal; Drive conserva el avance en `Mi unidad/detector_eeg`. El límite de caché es 24 GiB con una reserva adicional para trabajar en sesiones de aproximadamente 51 GB de RAM.
5. `prepare` genera características. `pilot` comprueba aprendizaje en TRAIN y descarta sus pesos. `train` completa las cinco rondas. `evaluate` calcula las pruebas después de fijar los cinco modelos. `verify` es una comprobación opcional de las selecciones desde probabilidades guardadas.
6. Conserva `avance.zip` y `diagnostico.zip` en Drive. Reanuda con este mismo cuaderno. No mezcles respaldos de otras versiones ni cambies reglas durante una ejecución.

## Qué se puede reproducir

Se puede volver a ejecutar el método y verificar las tablas publicadas. El código científico del ZIP es el original, sin ajustes para mejorar cifras después de la prueba. Entrenar otra vez puede producir diferencias por entorno y GPU. El repositorio no aporta los grandes archivos de probabilidades por época ni las matrices EEG remotas. Los pesos recuperados no se distribuyen como si su identidad fuese la del diagnóstico original.

La preferencia por el modelo de 114 detecciones frente al ajuste posterior de 109 se decidió tras conocer prueba. Las cinco pruebas de cada corrida se fijaron y reportaron sin seleccionar los mejores pacientes de modelos diferentes. Esto no elimina la exposición histórica a los resultados.

## Alcance

El detector identifica actividad durante una crisis. No predice una crisis antes de su inicio. No se debe utilizar para tomar decisiones médicas. Los PDF de los autores y los EDF originales se obtienen de sus fuentes y no se redistribuyen aquí. La publicación del repositorio no equivale a una revisión clínica independiente.
