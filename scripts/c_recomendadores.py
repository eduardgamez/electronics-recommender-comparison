import abc
import math
import pickle

class BaseRecommender(abc.ABC):
    """
    Clase base abstracta para todos los modelos de recomendación.
    Define la interfaz estándar que todos los modelos (KNN, PMF, BeMF, NCF) deben implementar.
    """
    
    @abc.abstractmethod
    def predict(self, u, i):
        """
        Calcula la predicción de la preferencia del usuario 'u' sobre el ítem 'i'.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load_model(cls, path):
        """
        Carga el estado del modelo desde la ruta especificada.
        """
        pass


class KNNPredictor(BaseRecommender):
    """
    Implementación del recomendador K-Nearest Neighbors (Filtrado Colaborativo).
    """
    
    def __init__(self, train_dict, k=50, similarity_metric='pearson', aggregation='weighted_average'):
        """
        Inicializa el modelo KNN.
        
        Parámetros:
        - train_dict (dict): Diccionario anidado de votos {usuario: {item: rating}}
        - k (int): Número máximo de vecinos a considerar.
        - similarity_metric (str): Métrica de similitud ('pearson' o 'jmsd').
        - aggregation (str): Estrategia para agregar la información.
        """
        self._train_dict = train_dict
        self._k = k
        self._similarity_metric = similarity_metric
        self._aggregation = aggregation
        
        # Caché de medias de usuario para mejorar la eficiencia general
        self._user_averages = {}
        for user in self._train_dict:
            self._user_averages[user] = self._rating_average(user)
            
    def _rating_average(self, u):
        """
        Calcula la nota media que ha dado el usuario 'u' a todos los ítems en el conjunto de entrenamiento.
        """
        if u not in self._train_dict or len(self._train_dict[u]) == 0:
            return 0.0
        ratings = list(self._train_dict[u].values())
        return sum(ratings) / len(ratings)
        
    def _correlation_similarity(self, u, v):
        """
        Calcula la similitud de correlación de Pearson entre los usuarios 'u' y 'v'.
        """
        # Encontramos los elementos que ambos usuarios han votado
        common_items = set(self._train_dict.get(u, {})).intersection(set(self._train_dict.get(v, {})))
        if not common_items:
            return 0.0
            
        avg_u = self._user_averages.get(u, 0.0)
        avg_v = self._user_averages.get(v, 0.0)
        
        num = 0.0
        dem_u = 0.0
        dem_v = 0.0
        
        for item in common_items:
            r_ui = self._train_dict[u][item] - avg_u
            r_vi = self._train_dict[v][item] - avg_v
            
            num += r_ui * r_vi
            dem_u += r_ui ** 2
            dem_v += r_vi ** 2
            
        if dem_u == 0 or dem_v == 0:
            return 0.0
            
        return num / (math.sqrt(dem_u) * math.sqrt(dem_v))
        
    def _jmsd_similarity(self, u, v):
        """
        Calcula la similitud basada en JMSD (Jaccard Mean Squared Difference) 
        entre los usuarios 'u' y 'v' combinando correlación cualitativa y cuantitativa.
        """
        items_u = set(self._train_dict.get(u, {}))
        items_v = set(self._train_dict.get(v, {}))
        
        common_items = items_u.intersection(items_v)
        union_items = items_u.union(items_v)
        
        if not common_items or not union_items:
            return 0.0
            
        # 1. Coeficiente de Jaccard (qué proporción de ítems comparten)
        jaccard = len(common_items) / len(union_items)
        
        # 2. Diferencia Media Cuadrada (MSD) sobre ítems co-votados
        sum_sq_diff = 0.0
        for item in common_items:
            diff = self._train_dict[u][item] - self._train_dict[v][item]
            sum_sq_diff += diff ** 2
            
        msd = sum_sq_diff / len(common_items)
        
        # Max diferencia posible al cuadrado para normalizar (ratings de 1 a 5: diff max = 4 -> sq = 16)
        max_sq_diff = 16.0 
        
        # La similitud JMSD es la mezcla de ambos
        jmsd = jaccard * (1 - (msd / max_sq_diff))
        
        return max(0.0, jmsd)
        
    def _get_neighbors(self, u, i):
        """
        Encuentra y ordena los 'k' vecinos más similares al usuario 'u' que SÍ hayan valorado el ítem 'i'.
        """
        neighbors = []
        for v in self._train_dict:
            # Solo consideramos usuarios distintos que hayan interactuado con el ítem 'i'
            if v != u and i in self._train_dict[v]:
                if self._similarity_metric == 'pearson':
                    sim = self._correlation_similarity(u, v)
                elif self._similarity_metric == 'jmsd':
                    sim = self._jmsd_similarity(u, v)
                else:
                    sim = 0.0
                    
                # Nos quedamos con los vecinos que aportan información positiva
                if sim > 0:
                    neighbors.append((v, sim))
                    
        # Ordenamos los vecinos por similitud descendente y aplicamos el corte K
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:self._k]

    def _compute_prediction(self, u, i):
        """
        Computa la predicción final basándose en la información ponderada de los K vecinos.
        """
        neighbors = self._get_neighbors(u, i)
        avg_u = self._user_averages.get(u, 0.0)
        
        # Si no hay vecinos válidos, devolvemos un baseline (la media histórica del usuario)
        if not neighbors:
            return avg_u 
            
        num = 0.0
        dem = 0.0
        
        for v, sim in neighbors:
            avg_v = self._user_averages.get(v, 0.0)
            r_vi = self._train_dict[v][i]
            
            # Estimación centralizada para evitar sesgos de "usuarios duros vs amables"
            num += sim * (r_vi - avg_v)
            dem += sim
            
        if dem == 0:
            return avg_u
            
        # Acoplamos el desplazamiento a la media del usuario objetivo
        pred = avg_u + (num / dem)
        
        # Garantizamos que la predicción siempre caiga en el espacio de votos válido (1 a 5)
        return min(max(pred, 1.0), 5.0)

    def predict(self, u, i):
        """
        Método público, invoca la tubería privada de predicción de la clase.
        """
        return self._compute_prediction(u, i)

    @classmethod
    def load_model(cls, path):
        """
        Método público de clase, recupera un objeto KNN guardado con serialización de bytes.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
