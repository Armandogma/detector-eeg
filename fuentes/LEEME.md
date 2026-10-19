# Corrección del piloto para cuDNN

Revisión de software `pilot_cudnn_train_mode_1`, misma familia científica
`tcn_gru_breves_cv_v1`. Corrige un fallo de ejecución. No cambia personas, particiones,
características, arquitectura, pérdidas, semillas, 60 pasos del piloto o criterios.
El piloto original medía la pérdida con model.eval() y omitía model.train() antes del
backward. La GRU de cuDNN necesita el forward de aprendizaje en modo train. Ahora se activa
train antes de los 60 pasos y eval al medir la pérdida final. El dropout del piloto sigue
siendo cero. Los pesos se descartan como antes.

Abre `Colab_Breves_CV_Corregido.ipynb` en la MISMA sesión de Colab con GPU y sube
`Paquete_Colab_Breves_CV_Corregido.zip`. Ejecuta desde la primera celda.
La base SSD se verifica y reutiliza. Si siguen disponibles las 590 matrices anteriores,
se enlazan y verifican por SHA-256 sin duplicarlas o calcularlas otra vez. prepare deberá
leerlas para comprobarlas, lo cual tarda un poco. Si la sesión se borró, habrá que recuperar
la base y recalcular las características; no se promete que el SSD temporal sobreviva.

Trabajo nuevo `/content/tesis_breves_cv_corregido` y respaldo separado
`/content/drive/MyDrive/tesis_breves_cv_corregido`. La carpeta original y su avance quedan
intactos. No se importan pilotos, pesos, selecciones o predicciones anteriores. El nuevo
manifiesto declara esta corrección y la huella del paquete original. No edites los hashes
ni copies avances entre ambas carpetas. No es necesario modificar a mano el archivo fuente.

La corrección debe verificarse con un piloto artificial completo en GPU y cuDNN, además
de reproducir el fallo anterior. Esto comprueba ejecución, no detección EEG ni eficacia.
El historial técnico siguiente describe el mismo protocolo científico. Sus rutas y nombres
del primer intento deben interpretarse con las rutas de la revisión indicadas arriba.

---

# Crisis breves y validación cruzada por persona

Este es un experimento nuevo para ejecutar en Colab con GPU. Usa los EEG reales en EDF de
`Mi unidad/base de datos`. Las señales artificiales se usan solamente para comprobar el
programa antes de entregarlo. Todavía no hay resultados EEG de esta versión.

El modelo anterior Adaptativo detectó 20/24 crisis en validación y 18/31 en prueba. Se conserva
ese resultado. Este paquete no cambia sus umbrales, pesos, archivos o pacientes.

## Qué se estudiará

Se conserva la familia TCN–GRU. La hipótesis nueva combina dos lecturas de la misma señal.
Una resume los cuatro segundos ya adquiridos y conserva el extractor de Adaptativo. La otra
resume los dos segundos recientes y alimenta un clasificador rápido sin memoria recurrente.
El modelo aprende a combinar ambas puntuaciones. Recibe 48 características por escala,
96 en total por derivación. No usa muestras futuras ni etiquetas para crear las entradas.

Cada escala describe ocho bandas entre 1 y 40 Hz. Por banda hay cinco medidas. Se añaden
dos contrastes rápido/lento y seis medidas de forma de la señal. El fondo de la rama de
dos segundos se calcula solo con ventanas anteriores que terminan antes o al comenzar la
ventana actual. Al principio del EDF se usa media cero y desviación uno si todavía no hay
pasado. La rama de cuatro segundos conserva la inicialización y normalización causal
anteriores. Los originales permanecen intactos. El filtro es causal de 0.5–40 Hz y la
limitación suave de amplitud se aplica al crear características, sin descartar horas.

La red temporal mantiene bloques TCN dilatados y GRU. Hay una salida adicional rápida sobre
las características de dos segundos. La salida final combina sus logits con un peso
aprendido. La pérdida usa 70% salida combinada, 15% rama temporal y 15% rama rápida.
Cada pérdida conserva el balance asimétrico anterior entre segundos positivos y negativos.

Durante TRAIN se elige primero una persona y después una de sus crisis con igual
probabilidad. Así una crisis larga no recibe automáticamente más oportunidades que una
corta. En la mitad de las elecciones positivas se busca el inicio del evento. Los primeros
cuatro segundos realmente etiquetados como crisis pesan el doble dentro de la media
positiva de la pérdida. No se cambia el inicio o final anotado ni se convierte el tiempo
anterior en crisis. Cada lote incluye una ancla positiva y dos secuencias negativas por
persona TRAIN. Se conservan aumentos de escala de características 0.85–1.15 solo TRAIN.

La minería de negativos difíciles se limita a TRAIN, en las épocas 4, 8 y 12. Conserva hasta
300 anclas por persona en cada actualización y comprueba que sus secuencias no incluyan
etiquetas positivas. Ahora están disponibles todas las horas TRAIN, no únicamente dos
registros negativos completos por persona. Estos cambios son una hipótesis integrada. Una
eventual diferencia no demostraría cuál de sus componentes causó una mejora.

## Cómo se separan las personas

Se conservan 20 personas pediátricas documentadas hasta 18 años inclusive, 590 EDF y 168
crisis elegibles. chb01 y chb21 son una sola persona y permanecen juntos. No se retiran
chb06, chb14, chb15 o chb16 por rendimiento. Se mantienen las exclusiones de edad y montaje
ya documentadas, sin añadir otras. No se usa SMOTE ni se crean EEG de crisis sintéticas.

Se reutiliza el orden de identidades de la CV ya entregada. No se buscan nuevas semillas.
Cada ronda entrena desde pesos nuevos con 12 personas, valida con cuatro y prueba con otras
cuatro. Cada persona aparece una vez en prueba externa y una en validación.

| Ronda | Crisis TRAIN | Crisis validación | Crisis prueba |
|---|---:|---:|---:|
| 1 | 113 | 24 | 31 |
| 2 | 119 | 25 | 24 |
| 3 | 75 | 68 | 25 |
| 4 | 80 | 20 | 68 |
| 5 | 117 | 31 | 20 |
| Total de pruebas externas | — | — | 168 |

En la ronda uno prueban chb01/chb21, chb07, chb10 y chb16. En las siguientes prueban los otros
grupos. Que una persona entrene un modelo de otra ronda es parte de la rotación. El modelo
que evalúa a esa persona nunca la usó para entrenarse o seleccionar su época o alarma.

Las características son una transformación fija de cada EDF y su propio pasado. Por eso
se preparan una vez los 590 EDF y se comparten entre rondas. No se ajusta una escala global
con todas las personas ni se mezclan sus señales. Las etiquetas entran al muestreador
exclusivamente según el papel TRAIN de cada ronda. Validación y prueba conservan todas sus
horas elegibles. No se evalúan únicamente recortes escogidos alrededor de crisis.

## Elección de época y de alarma

Hay un piloto de un lote TRAIN por ronda. Debe reducir la pérdida al menos 20% en 60 pasos.
Sus pesos se descartan. Un fallo detiene la ejecución y se conserva, sin repetir semillas
automáticamente. Luego cada ronda completa 15 épocas de 200 lotes de 36 secuencias de 24
pasos. AdamW usa tasa 0.0005 y decaimiento 0.001. El modelo no necesita esperar 24 segundos
para emitir cada puntuación. Los lotes de inferencia conservan causalidad y cubren cada
segundo desde la primera ventana completa de cuatro segundos, con reinicios de memoria
en una cuadrícula fija de bloques solapados.

Cada época se evalúa sobre la validación continua de su ronda. Se guardan sus probabilidades,
AP, punto operativo y clave de selección. Se prioriza cumplir los cinco objetivos. Después
se priorizan sensibilidad media por persona, sensibilidad conjunta, AP media y menores falsas
alarmas y tiempo bajo alarma. Un empate exacto conserva la época más temprana.

Los objetivos conservados son sensibilidad conjunta >=70%, media por persona >=60%, al menos
una detección por persona, <=1 falsa alarma por hora sin crisis y <=5% del tiempo bajo alarma.
El límite de falsas alarmas es global, no una garantía para cada persona.

La búsqueda de alarma queda fijada antes de entrenar. Incluye medias causales de 1, 3 o 5 s
y aperturas de dos o tres pasos, junto con los umbrales y cuantiles declarados en el JSON.
Solo se eligen con validación. No hay un umbral especial para chb16 ni para otros pacientes.

**Diferencia explícita con el ensayo fijo anterior:** en CV se fijan los cinco modelos y sus
reglas antes de abrir cualquier prueba externa. Se evalúan y entregan las cinco rondas,
incluidas las que no alcancen los objetivos en validación. Esos fallos quedan identificados.
No se ocultan rondas para aumentar el resultado global. Esta política no reabre ni cambia los
rechazos históricos. No se elige otro método al observar prueba.

## Qué resultados mostrará

El resultado principal será cuántas de las 168 crisis se detectaron con el modelo reservado
para cada persona. Se informarán sensibilidad por crisis, falsas alarmas/h, precisión y F1
por evento, tiempo bajo alarma, cobertura y demora solo entre detectadas. También habrá
resultados por ronda, por persona y por duración del evento. Las medias entre personas o
rondas son resúmenes distintos de la fracción conjunta de crisis detectadas.

Se cuenta solapamiento positivo uno a uno, sin tolerancia, sin fusionar crisis ni dividirlas
para ganar detecciones. Una alarma no detecta varias crisis. Siempre se guarda el conteo
alternativo de inicio de alarma dentro de la crisis y el tiempo completo bajo alarma. Una
alarma que ya estaba activa al inicio no demuestra predicción anticipada. No es SzCORE
completo. AP/AUROC y métricas por segundo se guardan aparte de la sensibilidad por crisis.

Las siete figuras son aprendizaje, validación, personas, grupos, demoras, duraciones y
auditoría de alarmas. Los intervalos por remuestreo de personas sobre predicciones fijas no
incorporan volver a entrenar las redes.

## Cómo ejecutarlo en Colab

1. Abre `Colab_Breves_CV.ipynb` y activa GPU en el entorno de ejecución.
2. Ejecuta desde la primera celda. Conecta Drive y sube `Paquete_Colab_Breves_CV.zip` cuando
   lo solicite. No se requiere subir EDF desde la laptop.
3. El cuaderno verifica el SSD `/content/cache_pediatrica_ssd`. Si falta algo, busca la base
   de Drive en `Mi unidad/base de datos` y copia solo los archivos pendientes.
4. Ejecuta preparación, pilotos, entrenamiento y prueba cruzada en ese orden. La prueba
   comprueba que los cinco modelos y selecciones estén congelados.
5. Los resultados y `avance.zip` quedan en `Mi unidad/tesis_breves_cv`. La última celda crea
   `diagnostico.zip` aunque falle la descarga del navegador. El respaldo incluye pesos,
   optimizador, aleatoriedad, minería y cada época completa. El diagnóstico no lleva pesos.

Trabajo temporal `/content/tesis_breves_cv`. Esta versión no comparte avances o pesos con
`tesis_adaptativa`, `tesis_relativa` u otras familias. La caché de características ocupa unos
8.96 GiB. EDF elegibles y características nuevos requieren unos 42.82 GiB antes de resultados
y margen, sin borrar otras carpetas. La caché RAM tiene tope 24 GiB y reserva 8 GiB; dejar
memoria libre permite trabajar con unos 51 GB sin intentar ocuparlos todos. La preparación
usa dos trabajadores. La inferencia construye solo el lote actual, no todas las ventanas
del EDF en RAM a la vez. Más RAM puede acelerar lectura, no asegura mayor sensibilidad.

Si Colab se reinicia, abre este mismo cuaderno y ejecuta desde arriba. Recupera el último
respaldo íntegro de esta familia. Si perdió las características temporales, se regeneran
con los mismos EDF. Se comprueban versiones y GPU para evitar mezclar avances incompatibles.
No cambies el JSON o las fuentes durante una corrida. `verify` permite reproducir todas las
selecciones interépoca desde sus probabilidades de validación, sin buscar cortes en prueba.

## Fundamento y límites

[Dan y colaboradores, SzCORE](https://epilepsybenchmarks.com/framework/) describen CV por
persona y separación de sujetos para estudiar generalización. También recomiendan distinguir
métricas por muestras y por eventos. La división fija anterior era válida; CV estudia ahora
la variación al cambiar las personas reservadas. Nuestros parámetros y conteos conservan
las reglas propias descritas arriba y no reproducen todo el marco SzCORE.

[Ingolfsson y colaboradores, 2024](https://www.nature.com/articles/s41598-024-52551-0)
estudiaron ventanas y la relación entre detección y falsas alarmas en EEG para dispositivos
portátiles. Sus clasificadores y evaluación difieren de este paquete. El trabajo motiva
examinar la duración de las ventanas, pero no demuestra que 2 s sea óptimo aquí ni que
nuestros pacientes sean ruido. La combinación 2+4 s, el cabezal rápido y los pesos de inicio
son hipótesis propias, no una reproducción de ese artículo.

El método se propone después de observar errores anteriores, incluidos chb16 y chb14. Toda
esta evaluación es exploratoria por la exposición histórica a prueba y las consultas
repetidas a validación. CV no borra esa exposición. No hay garantía de detectar 130 o 140
crisis ni de utilidad clínica. Se conservarán todos los resultados aunque sean desfavorables.
