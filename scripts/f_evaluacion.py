# =====================================================================
# Módulo Global de Funciones de Evaluación
# Todas estas funciones reciben un modelo que cumple con BaseRecommender
# =====================================================================

import math

def get_rmse(test_dict, model):
    """
    Calcula el Root Mean Squared Error (RMSE). 
    Es la métrica más estándar: penaliza fuertemente los errores grandes de predicción.
    """
    squared_errors = []
    for u in test_dict:
        for i, true_rating in test_dict[u].items():
            pred_rating = model.predict(u, i)
            squared_errors.append((true_rating - pred_rating) ** 2)
            
    if not squared_errors:
        return 0.0
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def get_mae(test_dict, model):
    """
    Calcula el Mean Absolute Error (MAE).
    Mide el error absoluto lineal en la misma escala que los ratings.
    """
    abs_errors = []
    for u in test_dict:
        for i, true_rating in test_dict[u].items():
            pred_rating = model.predict(u, i)
            abs_errors.append(abs(true_rating - pred_rating))
            
    if not abs_errors:
        return 0.0
    return sum(abs_errors) / len(abs_errors)


def get_precision(test_dict, model, threshold=3.5):
    """
    Calcula la Precisión: proporción de ítems recomendados por el modelo 
    que de verdad gustaron al usuario en test.
    """
    relevant_recommended = 0
    total_recommended = 0
    
    for u in test_dict:
        for i, true_rating in test_dict[u].items():
            pred_rating = model.predict(u, i)
            
            if pred_rating >= threshold:
                total_recommended += 1
                if true_rating >= threshold:
                    relevant_recommended += 1
                    
    if total_recommended == 0:
        return 0.0
    return relevant_recommended / total_recommended


def get_recall(test_dict, model, threshold=3.5):
    """
    Calcula el Recall (Exhaustividad): proporción de todos los ítems reales 
    que gustaron al usuario y que el modelo logró recomendarle.
    """
    relevant_recommended = 0
    total_relevant = 0
    
    for u in test_dict:
        for i, true_rating in test_dict[u].items():
            pred_rating = model.predict(u, i)
            
            if true_rating >= threshold:
                total_relevant += 1
                if pred_rating >= threshold:
                    relevant_recommended += 1
                    
    if total_relevant == 0:
        return 0.0
    return relevant_recommended / total_relevant


def get_ndcg(test_dict, model, k=10):
    """
    Calcula el Normalized Discounted Cumulative Gain (nDCG).
    Evalúa la eficacia para rankear listas Top-K: un modelo es bueno si 
    los productos con notas más altas quedan en las primeras posiciones.
    """
    ndcg_sum = 0.0
    users_count = 0
    
    for u, items in test_dict.items():
        if len(items) < 2:
            continue
            
        # Generar las predicciones del modelo para todos los ítems reales en el test
        predictions = [(i, model.predict(u, i)) for i in items]
        
        # Ordenación ideal (la real del usuario) frente a ordenación del modelo
        ideal_order = sorted([(i, r) for i, r in items.items()], key=lambda x: x[1], reverse=True)
        pred_order = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        ideal_top_k = ideal_order[:k]
        pred_top_k = pred_order[:k]
        
        # Descuento Acumulado Ideal (IDCG)
        idcg = 0.0
        for rank, (item, rating) in enumerate(ideal_top_k, 1):
            idcg += (2 ** rating - 1) / math.log2(rank + 1)
            
        # Descuento Acumulado del Modelo (DCG)
        dcg = 0.0
        for rank, (item, _) in enumerate(pred_top_k, 1):
            true_rating = items.get(item, 0.0)
            dcg += (2 ** true_rating - 1) / math.log2(rank + 1)
            
        if idcg > 0:
            ndcg_sum += dcg / idcg
        users_count += 1
        
    if users_count == 0:
        return 0.0
    return ndcg_sum / users_count
