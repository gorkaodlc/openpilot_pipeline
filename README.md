# OPENPILOT SUPERCOMBO MODEL
### INFERENCIA
La inferencia se hace en el siguiente script 
```
inference/openpilot_realtime_inf.py
```

### EVALUACIÓN
La evaluación se hace en tiempo real, donde existe un diagrama de RTMaps que teniendo un RECORD y la trayectoria Ground Truth real, hace la inferencia y extrae metricas para cada detección. El flujo tiene que ser el siguiente:
0. Partimos de un video con resolución (1160, 874) grabado con la camara lo mas en paralelo con el suelo posible
1. Extraer el path en X, Y, Z del GPS y generar un nuevo record.
```
path_extraction/gps_preprocess/preprocess.rtd
```
2. Inferencia del modelo + extracción de metricas.<br>
**RTMaps**
```
rtmaps/inference_from_record.rtd
```
**Python script**
```
inference/openpilot_realtime_inf.py
```