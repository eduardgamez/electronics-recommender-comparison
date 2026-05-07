# Amazon Electronics Recommendation Engine
**Personalization Benchmark Suite** que implementa y compara cuatro paradigmas de **Filtrado Colaborativo** para automatizar la recomendación de productos de electrónica.

## Descripción Técnica
* **Arquitectura:** Framework de evaluación comparativa diseñado para contrastar 4 métodos: KNN (k-Nearest Neighbors), PMF (Probabilistic Matrix Factorization), BeMF (Bayesian Estimated Marginalized Filters) y NCF (Neural Collaborative Filtering).
* **Flujo de Datos:**
  * **Origen:** Dataset real de Amazon Electronics con interacciones de usuario. *Procesado mediante `python dataset/process_data.py`*.
  * **Dimensiones:** Vectores de interacción simbólica: User ID, Item ID, Rating (1-5) y Timestamp.
  * **Target:** Estimación de la preferencia para la optimización de listas de recomendación Top-N.
* **Objetivo final:** Optimizar la precisión del sistema comparando la capacidad de generalización de los modelos frente a un "Baseline" (media global).

## Organización del Proyecto
* `dataset/`: Contiene el pipeline de limpieza y las particiones de datos resultantes (`train.csv`, `val.csv`, `test.csv`).
* `scripts/`: El núcleo del motor.
  * `functions.py`: Implementación de las clases de los 4 modelos (KNN, PMF, BeMF, NCF) y lógica de entrenamiento.
  * `tuning_knn.ipynb`: Optimización de $k$ vecinos y métricas de similitud (Pearson/JMSD).
  * `tuning_pmf.ipynb`: Ajuste de factores latentes y regularización $\lambda$.
  * `tuning_bemf.ipynb`: Modelado de distribuciones de Bernoulli y fiabilidad.
  * `tuning_ncf.ipynb`: Entrenamiento de redes profundas (GMF/MLP) en PyTorch.
* `models/`: Almacén de subproductos, como pesos de redes neuronales y matrices de factores entrenadas.
* `comparison/`: `final_comparison.ipynb`. Notebook de ejecución final que contrasta el rendimiento de los mejores candidatos de cada técnica.

## Flujo de Ejecución (Pipeline)
Para garantizar la máxima precisión en un entorno de alta dimensionalidad, el sistema sigue tres fases:

1.  **Refinado de Datos**: Se filtran usuarios e ítems con baja actividad para reducir la dispersión de la matriz de votaciones, generando particiones consistentes para validación cruzada. Se puede ejecutar el script de procesamiento mediante `python dataset/process_data.py` para generar los archivos CSV limpios.
2.  **Ajuste de Hiperparámetros (Tuning)**: Cada notebook en `scripts/` ejecuta una búsqueda para encontrar el equilibrio entre sesgo y varianza. Se optimizan parámetros críticos como el número de factores latentes o el *learning rate*.
3.  **Evaluación Multidimensional**: Los modelos se enfrentan en la carpeta `comparison/`. No solo se busca minimizar el $RMSE$, sino maximizar el **nDCG**, asegurando que los productos preferidos por el usuario aparezcan en las primeras posiciones de la lista.

## Conclusión: 
...

## Instalación de dependencias
Escribe estos comandos uno a uno en la terminal de tu editor para instalar las dependencias:

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
