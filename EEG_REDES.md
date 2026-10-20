# EEG Redes

## Qué permite hacer este proyecto

Este proyecto estudia cómo detectar una crisis epiléptica mientras ocurre en un registro de EEG (*electroencephalogram*). El cuaderno de Colab lee las señales originales, prepara medidas de esas señales, entrena una red y compara sus alarmas con las crisis anotadas por la fuente de datos.

El resultado histórico fue **114 de 168 crisis detectadas**, sensibilidad de **67.86%** y **1.216 falsas alarmas por hora sin crisis**. La demora mediana fue de 11 segundos entre las crisis detectadas. Son resultados exploratorios. Una nueva ejecución puede dar cifras diferentes y este código no tiene validación para tomar decisiones médicas. Detectar una crisis que ya comenzó es distinto de predecirla antes de su inicio.

La red final combina dos ramas. Una analiza medidas de los últimos 4 segundos y usa una red convolucional temporal, TCN (*temporal convolutional network*), y una unidad de memoria, GRU (*gated recurrent unit*). La otra analiza medidas de los últimos 2 segundos sin memoria recurrente. Ambas salidas se combinan para generar una probabilidad de crisis cada segundo. Este resultado corresponde a esa red, no a los modelos CNN–BiLSTM–SupCon de los ensayos anteriores.

## Qué datos se necesitan

Se utilizó [CHB-MIT versión 1.0.0, publicada en PhysioNet](https://physionet.org/content/chbmit/1.0.0/). Contiene registros de pacientes con epilepsia resistente al tratamiento atendidos en el Children's Hospital Boston. Las carpetas, como `chb01`, identifican casos. Una carpeta puede contener muchos archivos de distintas horas o días. `chb01` y `chb21` corresponden a la misma persona y se mantienen juntas.

Los archivos EDF (*European data format*) contienen varias señales registradas al mismo tiempo. Generalmente duran entre una y cuatro horas. La frecuencia es 256 mediciones por segundo y la resolución es de 16 bits. Cada señal EEG describe una medición eléctrica o una diferencia entre electrodos. El número y nombre de las señales pueden variar entre archivos. El procedimiento usa 18 derivaciones bipolares que puede reconstruir de manera verificable.

Los archivos `chbXX-summary.txt` describen los registros, las derivaciones y los segundos de inicio y fin de las crisis dentro de cada EDF. Las anotaciones adicionales también se comprueban cuando están disponibles. Esos tiempos sirven para construir las etiquetas de referencia. El modelo debe aprender a reconocer patrones de la señal, no recibir los tiempos de las crisis al analizar la prueba.

La cohorte del experimento comprende **20 personas de 1.5 a 18 años, 590 EDF, 773.64 horas registradas y 168 crisis**. No representa toda la colección. Se excluyeron `chb04` y `chb19` por edad mayor de 18 años y `chb24` porque su edad no constaba en la fuente de edades usada. De los 592 EDF de edad elegible se apartaron `chb12_28` y `chb12_29`, con siete crisis, por una referencia y una etiqueta de canal que no permitían reconstruir el montaje elegido sin ambigüedad. No se excluyeron personas por detectar pocas crisis.

Los EDF y los artículos de otros autores no se incluyen en este repositorio. Se obtienen de sus fuentes y se respetan sus licencias. Para este experimento se necesitan las señales originales y sus anotaciones, no solamente un CSV de etiquetas.

## Cómo ejecutarlo en Colab

1. Coloca las carpetas `chb01`, `chb02` y las demás en `Mi unidad/base de datos` de Google Drive. Conserva los EDF, los resúmenes, las anotaciones y `SUBJECT-INFO` cuando esté disponible. Una ruta de ejemplo es `/content/drive/MyDrive/base de datos/chb01`.
2. [Abre el cuaderno completo en Colab](https://colab.research.google.com/github/Armandogma/detector-eeg/blob/main/Colab_Detector_EEG.ipynb) y selecciona un entorno con GPU. No necesitas crear otro programa ni subir manualmente el ZIP científico. El cuaderno lo descarga del repositorio y comprueba que corresponde a la versión original.
3. Ejecuta las celdas en orden. La primera monta Drive y define las carpetas. Si tu base está en otro sitio, corrige únicamente la ruta `DRIVE_DATASET` antes de continuar.
4. Deja que copie los EDF necesarios al SSD temporal y prepare las características. La copia no elimina la base de Drive. Aunque tienes unos 51 GB de RAM, el protocolo fija una caché máxima de 24 GiB y reserva al menos 8 GiB para el resto del trabajo. No es necesario cargar toda la base a la vez.
5. Ejecuta el piloto, el entrenamiento y la evaluación en las celdas correspondientes. El piloto verifica que el programa puede aprender un lote de entrenamiento y después descarta esos pesos. No mide la eficacia del detector.
6. Conserva en Drive `Mi unidad/detector_eeg/avance.zip` y `diagnostico.zip`. El primero guarda los pesos y la información para reanudar. El segundo reúne información para revisar resultados. Pueden ocupar bastante espacio. Si una descarga falla, comprueba primero si el archivo quedó en Drive.

El SSD de Colab se borra al terminar o reiniciarse la sesión. Drive conserva los archivos que sí se hayan guardado allí. Al volver, usa el mismo cuaderno y el respaldo de esa misma versión. Puede ser necesario copiar otra vez los EDF y regenerar las características. Se recuperan épocas completas, no los lotes incompletos. Un cambio de GPU o de versiones puede impedir continuar exactamente la corrida anterior. No mezcles respaldos de experimentos diferentes.

## Cómo se entrena y se pone a prueba

Se forman cinco grupos de cuatro personas. En cada ronda se usan doce personas para entrenar, cuatro para validar y cuatro para probar. Cada persona aparece una sola vez en la prueba final. Se suman esas cinco pruebas para obtener el resultado sobre las 168 crisis. Las crisis de validación no se suman a ese total.

El entrenamiento utiliza secuencias seleccionadas de los registros de entrenamiento. Incluye crisis, momentos de inicio y periodos sin crisis. No recorre todos los segundos en cada época. Cada ronda completa 15 épocas de 200 lotes. Una época es una vuelta por ese conjunto de lotes, no por toda la colección. La validación y la prueba recorren los registros continuos disponibles de sus personas, incluidas las largas horas sin crisis. De ese modo se puede medir cuántas falsas alarmas genera el sistema durante la vigilancia.

La validación elige la época y la regla que transforma probabilidades en alarmas. Se consideran sensibilidad, diferencias entre personas, falsas alarmas y tiempo marcado. Los cinco modelos y sus reglas se fijan antes de calcular las pruebas de esa corrida. Se informa el resultado de todas las rondas, incluso cuando una no alcanza los objetivos de validación. Estos datos ya se consultaron en ensayos anteriores, por lo que el estudio se presenta como exploratorio.

El filtro usa la banda de 0.5 a 40 Hz. La preparación calcula características de 2 y 4 segundos y referencias a partir del pasado del mismo registro. No usa muestras futuras ni mezcla estadísticas de prueba entre pacientes para normalizar. El filtro y la normalización reducen variaciones de la señal, pero no demuestran que toda actividad extraña sea ruido ni garantizan eliminar las falsas alarmas.

## Para qué sirve cada archivo principal

Los enlaces permiten leer los archivos sin descargar todo el proyecto.

| Archivo | Para qué sirve |
|---|---|
| [Colab_Detector_EEG.ipynb](Colab_Detector_EEG.ipynb) | Es el cuaderno que se ejecuta. Conecta Drive, descarga el código verificado y muestra las etapas y sus resultados. |
| [codigo_congelado.zip](codigo_congelado.zip) | Conserva el paquete científico original. El cuaderno comprueba su huella digital antes de extraerlo. |
| [fuentes/](fuentes/) | Muestra, para lectura, los mismos archivos incluidos en el ZIP. No es un segundo experimento. |
| [METODO.md](METODO.md) | Describe con más detalle las características, el entrenamiento y el cálculo de resultados. |
| [SHA256SUMS.json](SHA256SUMS.json) | Guarda las huellas digitales de los archivos. Permite detectar si cambiaron. |
| [verificar_entrega.py](verificar_entrega.py) | Revisa las huellas, las divisiones de pacientes y los recuentos históricos. No entrena ni analiza nuevas señales. |
| [fuentes/paquete_manifest.json](fuentes/paquete_manifest.json) | Identifica la versión científica y las huellas de sus fuentes. |
| [fuentes/LEEME.md](fuentes/LEEME.md) | Conserva las instrucciones escritas al entregar el paquete original. Sus frases sobre resultados pendientes se refieren a aquel momento. El resultado recibido después se describe en esta guía y en `resultados/`. |

## Qué hace cada archivo de código

No es necesario ejecutar uno por uno estos archivos. El cuaderno llama al programa principal y este utiliza los demás según los necesita.

| Archivo en `fuentes/codigo/` | Función en este experimento |
|---|---|
| [colab_breves_cv.py](fuentes/codigo/colab_breves_cv.py) | Dirige las cinco rondas. Prepara datos, hace el piloto, entrena, fija las selecciones y calcula las pruebas. Es el programa principal. |
| [configuracion_breves_cv.json](fuentes/codigo/configuracion_breves_cv.json) | Guarda los valores fijados para el experimento, como ventanas, filtro, número de épocas y reglas de alarma. Es un archivo de configuración, no una red. |
| [modelo_breves.py](fuentes/codigo/modelo_breves.py) | Define la red final de dos escalas, la rama rápida, la mezcla de salidas, el muestreo de entrenamiento y su función de pérdida. La pérdida indica al programa cuánto se equivoca para que ajuste sus pesos. |
| [modelo_adaptativo.py](fuentes/codigo/modelo_adaptativo.py) | Aporta las características de 4 segundos, las referencias al pasado y la rama TCN–GRU reutilizada por la red final. |
| [dataset_causal.py](fuentes/codigo/dataset_causal.py) | Lee las señales calibradas de los EDF y contiene herramientas para manejar canales y segmentos. No convierte toda la señal en una simple etiqueta. |
| [colab_adaptativo.py](fuentes/codigo/colab_adaptativo.py) | Aporta funciones comunes de lectura, respaldo, control de versiones, selección de alarmas y cálculo de resultados. El programa principal reutiliza estas funciones; no lanza por ello otro ensayo adaptativo. |
| [eventos_relativos.py](fuentes/codigo/eventos_relativos.py) | Compara las alarmas con las crisis anotadas y calcula resultados por evento y por segundo. Evita contar varias alarmas como varias detecciones de una misma crisis. |
| [evaluador_continuo.py](fuentes/codigo/evaluador_continuo.py) | Reúne resultados de registros y personas y calcula medidas globales e intervalos de confianza. |
| [ajustes_modelo.py](fuentes/codigo/ajustes_modelo.py) | Aporta el promedio causal y las reglas que abren y cierran alarmas. También contiene clases antiguas que no son la arquitectura final de este artículo. |
| [rutas.py](fuentes/codigo/rutas.py) | Define dónde buscar las carpetas del proyecto. El cuaderno proporciona la ubicación real de la base. |
| [modelo_cnn_bilstm.py](fuentes/codigo/modelo_cnn_bilstm.py) | Contiene una arquitectura y una pérdida contrastiva de ensayos anteriores. Permanece por las dependencias del paquete. Su presencia no significa que el modelo final use BiLSTM o SupCon. |
| [modelo_atencion.py](fuentes/codigo/modelo_atencion.py) | Conserva una variante anterior de CNN–BiLSTM con atención de canales. No se entrena como red principal de este cuaderno. |
| [modelo_memoria.py](fuentes/codigo/modelo_memoria.py) | Conserva herramientas y un modelo de secuencias de una etapa anterior. No es la red final que produjo las 114 detecciones. |
| [modelo_espectro.py](fuentes/codigo/modelo_espectro.py) | Conserva un modelo anterior basado en espectrogramas y funciones relacionadas. No sustituye a `BriefSeizureNet` en este ensayo. |
| [modelo_relativo.py](fuentes/codigo/modelo_relativo.py) | Conserva otra representación espectral de un ensayo anterior. Se mantiene entre los archivos del paquete original, sin entrenarla como modelo adicional. |

Algunos textos generales de la configuración provienen del programa anterior. Para las cinco rondas, la política efectiva está en `test_policy`, en el apartado `cv` y en `colab_breves_cv.py`: se fijan todos los modelos y se informan todas las pruebas, incluso las rondas con validación insuficiente. No se deben cambiar las fuentes para resolver esa diferencia de redacción histórica.

## Qué contienen los inventarios

| Archivo en `fuentes/analisis/` | Para qué sirve |
|---|---|
| [manifiesto_dataset.json](fuentes/analisis/manifiesto_dataset.json) | Lista los registros, su duración, canales, anotaciones y problemas encontrados. El inventario amplio es diferente de la cohorte finalmente elegida. |
| [cohorte_pediatrica.json](fuentes/analisis/cohorte_pediatrica.json) | Indica las personas y los EDF incluidos por edad y por requisitos técnicos. Guarda también las exclusiones y sus motivos. |
| [particiones_breves_cv.json](fuentes/analisis/particiones_breves_cv.json) | Fija qué personas entrenan, validan y prueban en cada una de las cinco rondas. |
| [particion_rf_fija.json](fuentes/analisis/particion_rf_fija.json) | Conserva el reparto previo del que deriva el orden de los grupos. No indica que el cuaderno actual entrene Random Forest. |
| [calidad_modelo.json](fuentes/analisis/calidad_modelo.json) | Identifica bloques en los que las 18 derivaciones permanecen exactamente constantes y que no se usan al muestrear entrenamiento. No recorta validación o prueba ni demuestra ruido clínico. |

## Cómo leer los resultados

| Archivo en `resultados/` | Contenido |
|---|---|
| [resumen.json](resultados/resumen.json) | Cifras globales históricas y límites de su interpretación. |
| [breves_cv_rondas.csv](resultados/breves_cv_rondas.csv) | Resultado de cada ronda y su selección. |
| [breves_cv_eventos_prueba.csv](resultados/breves_cv_eventos_prueba.csv) | Una fila por cada una de las 168 crisis de prueba y si fue detectada. |
| [crisis_por_duracion.csv](resultados/crisis_por_duracion.csv) | Recuentos según duración. Se detectaron 2/10 crisis de hasta 10 segundos, 16/45 de más de 10 a 30 segundos y 96/113 de más de 30 segundos. |

La sensibilidad cuenta qué proporción de crisis se detectó. Las falsas alarmas por hora cuentan alarmas no emparejadas con una crisis, divididas entre las horas sin crisis. La precisión por evento indica cuántas de todas las alarmas correspondieron a una crisis detectada. Una sensibilidad aceptable puede coexistir con muchas falsas alarmas. La cobertura indica cuánto tiempo registrado pudo evaluarse. La demora solo describe las crisis detectadas, no las perdidas.

## Qué se puede comprobar y qué no se garantiza

Después de descargar el repositorio, se puede ejecutar `python verificar_entrega.py` con Python 3 para revisar las huellas y los recuentos sin entrenar. El método completo se vuelve a ejecutar con el cuaderno y los EDF obtenidos de PhysioNet. El SHA-256 del paquete científico original es:

```text
0c5e937b18bfcd8c5359d3dffec3b555173f08fdf0d6c884817437dc00d1f63c
```

El repositorio no incluye las grandes matrices de características, todas las probabilidades remotas ni una distribución de pesos que certifique identidad con el diagnóstico original. Publicar el código permite repetir el procedimiento, pero no garantiza obtener exactamente los mismos pesos o cifras en otra GPU o versión de software.

Se probaron varias estrategias antes de elegir la versión del artículo. La preferencia por 114 detecciones frente al ajuste posterior de 109 se decidió después de conocer los resultados de prueba. No se mezclaron los mejores pacientes de distintos modelos. Esta historia limita la independencia de la evaluación y se conserva al interpretar los resultados.
