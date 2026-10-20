# EEG Redes

Código y cuaderno del detector de crisis epilépticas en EEG continuo. El modelo combina una red TCN–GRU de 4 segundos con una rama rápida de 2 segundos.

El resultado histórico fue **114/168 crisis detectadas**, sensibilidad de **67.86%** y **1.216 falsas alarmas por hora sin crisis**. La precisión por evento fue 10.85% y la demora mediana fue 11 segundos entre las crisis detectadas. El estudio es exploratorio y no demuestra utilidad clínica.

## Por dónde empezar

1. Lee [EEG_REDES.md](EEG_REDES.md). Explica la base de datos, los pasos en Colab, las cinco rondas, los resultados y para qué sirve cada archivo.
2. Coloca la base CHB-MIT en `Mi unidad/base de datos` de Drive.
3. [Abre el cuaderno completo en Colab](https://colab.research.google.com/github/Armandogma/detector-eeg/blob/main/Colab_Detector_EEG.ipynb), activa una GPU y ejecuta las celdas en orden. El cuaderno descarga el código y comprueba su versión.

La base se obtiene de [PhysioNet, CHB-MIT 1.0.0](https://physionet.org/content/chbmit/1.0.0/), DOI [10.13026/C2K01R](https://doi.org/10.13026/C2K01R), bajo su licencia Open Data Commons Attribution 1.0. El repositorio no incluye los EDF ni los PDF de otros autores.

## Qué encontrarás

| Archivo o carpeta | Contenido |
|---|---|
| [EEG_REDES.md](EEG_REDES.md) | Guía sencilla y explicación de cada programa. |
| [Colab_Detector_EEG.ipynb](Colab_Detector_EEG.ipynb) | Cuaderno completo para ejecutar el procedimiento. |
| [codigo_congelado.zip](codigo_congelado.zip) | Paquete científico original verificado antes de ejecutarse. |
| [fuentes/](fuentes/) | Los mismos archivos del ZIP, visibles para lectura. |
| [METODO.md](METODO.md) | Descripción más detallada del método. |
| [resultados/](resultados/) | Tablas históricas por ronda, crisis y duración. |
| [verificar_entrega.py](verificar_entrega.py) | Comprobación de archivos y recuentos sin entrenar. |

## Reproducibilidad

El cuaderno conserva el protocolo de cinco rondas y la cohorte de 20 personas, 590 EDF y 168 crisis. Usa las señales reales y evalúa registros continuos. Una sesión de unos 51 GB de RAM puede utilizarlo con la caché fijada de 24 GiB. El SSD de Colab es temporal. Los respaldos se guardan en `Mi unidad/detector_eeg`.

El código científico permanece sin cambios respecto al paquete original. Una nueva ejecución puede producir diferencias por GPU y versiones. No se incluyen las grandes matrices o todas las probabilidades históricas. La preferencia por el resultado de 114 detecciones frente al ajuste posterior de 109 se decidió después de conocer prueba. Esa exposición previa se declara y limita la interpretación del estudio.
