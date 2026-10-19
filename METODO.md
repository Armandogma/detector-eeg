# Datos y funcionamiento del experimento

## Cohorte

Se incluyeron personas con edad documentada de 0 a 18 años, inclusive, como definición operativa del estudio. Se excluyeron chb04 (22 años), chb19 (19 años) y chb24 (edad no documentada). chb01 y chb21 corresponden a una misma persona. Entre los registros de edad elegible se conservaron 590 EDF de 592. Los archivos chb12_28 y chb12_29 tenían referencia sin aclarar y una etiqueta 01 ambigua que impidió reconstruir inequívocamente las 18 derivaciones. Aportaban siete crisis conocidas, que se declararon excluidas; no se contabilizaron como detectadas o perdidas. Se conservó chb12_27 mediante reconstrucción calibrada de la referencia común. El inventario final reúne 20 personas, 773.6378 horas y 168 crisis. No representa toda la colección CHB-MIT.

## Señales y dos escalas

El código aplica un filtro Butterworth causal de 0.5–40 Hz y un acotamiento suave de amplitud. Calcula 48 características por derivación en cada escala, 96 en total, a partir de potencia, bandas, cambios temporales y medidas de forma. La referencia se obtiene del pasado disponible del mismo EDF, sin etiquetas. No hay un ajuste conjunto con futuros pacientes. Los valores iniciales de cada escala están fijados en el código. Los estados se reinician entre EDF y no se unen los huecos de grabación. Las señales originales y las etiquetas no se reescriben.

La rama de 4 s tiene capas compartidas 48→64→32, cuatro resúmenes entre derivaciones, proyección 128→96, cuatro bloques TCN y una GRU de 64 estados. Se suma una proyección 96→64 a la salida recurrente. La rama rápida de 2 s utiliza capas 48→48→24, tres resúmenes entre derivaciones y clasificación 72→48→1. Una mezcla aprendida combina ambas puntuaciones antes de la sigmoide. Hay 173189 parámetros. No usa BiLSTM, SupCon, Mamba o SMOTE en esta versión.

## Cinco rondas

Los grupos fijos tienen cuatro personas cada uno. La ronda i prueba el grupo i, valida el siguiente de manera circular y entrena con los otros tres. Cinco rondas permiten probar una vez a cada persona y entrenar con 12 en cada ronda. Es una decisión práctica, no una demostración de que cinco sea el número óptimo. Se mantuvo el orden de una partición anterior sin buscar semillas por desempeño. En cada ronda hay 12 pacientes de entrenamiento, cuatro de validación y cuatro de prueba. La reserva interna no es una búsqueda anidada exhaustiva.

La prueba de cada ronda reúne 31, 24, 25, 68 y 20 crisis. La suma es 168 porque cada persona y EDF se probaron una sola vez con el modelo que los tenía reservados. No se suman las crisis de validación a la prueba. Aunque las características por EDF se puedan reutilizar, los pesos del modelo no se comparten entre rondas.

## Entrenamiento y mejor época

Cada ronda completa 15 épocas de 200 lotes. Una época aquí es el conjunto de 200 lotes muestreados, no una lectura exhaustiva de todos los segundos disponibles. Cada lote toma de cada paciente TRAIN un fragmento que contiene una crisis y dos sin crisis. Son 36 secuencias de 24 pasos. Todo el fondo TRAIN queda accesible al muestreo. La mitad de las anclas positivas se orienta al inicio; se da peso doble solo a los primeros cuatro segundos realmente positivos. Se buscan negativos difíciles solamente en TRAIN en las épocas 4, 8 y 12. La pérdida de la salida combinada pesa 70% y las auxiliares 15% cada una. Dentro de cada salida, positivos y negativos pesan 30/70. La variante posterior que normalizó por paciente es un experimento distinto.

Después de cada época se calcula validación completa y se elige una regla entre las prefijadas. La época elegida prioriza, por orden, cumplimiento de objetivos, sensibilidad media entre pacientes, sensibilidad agregada, precisión promedio, menos falsas alarmas y menos tiempo bajo alarma. En un empate exacto se conserva la época anterior. Por eso la mejor época no tiene que ser la última ni la de menor pérdida TRAIN. Se eligieron las épocas 4, 5, 4, 10 y 14. La segunda ronda no alcanzó el objetivo de validación, pero se reportó porque la política de esta CV exige informar las cinco pruebas.

## Evaluación

Todas las horas de validación y prueba se conservan, incluidos registros sin crisis, actividad interictal y artefactos no descartados por rendimiento. Se contabiliza una correspondencia uno a uno por solapamiento positivo sin tolerancia añadida. Se informa aparte el criterio de inicio de alarma dentro de crisis. No se afirma implementar SzCORE completo.

La sensibilidad es detectadas/168. La precisión por evento es 114/1051. Las falsas alarmas por hora usan 937 alarmas no emparejadas y las horas evaluadas sin crisis. Entre ellas hay 23 alarmas que se solapan con una crisis ya emparejada y 914 que no se solapan con ninguna. La cobertura usa tiempo evaluado/registrado. El área ROC y la precisión promedio se calculan por segundo y luego se promedian con igual peso por paciente. No sustituyen la sensibilidad por crisis. La demora solo resume las crisis detectadas. El remuestreo por persona no incorpora variabilidad de nuevos entrenamientos.

## Límites de reproducción y del resultado

Las tablas son verificables a partir de los archivos públicos de resultados de este repositorio y el protocolo se puede volver a ejecutar. Esto no garantiza obtener los mismos pesos ni certifica la procedencia de los archivos remotos por una tercera parte. No se han realizado pruebas clínicas prospectivas. Los experimentos previos y la preferencia posterior al resultado hacen que el estudio siga siendo exploratorio.
