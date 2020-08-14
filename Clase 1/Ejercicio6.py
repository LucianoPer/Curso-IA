import numpy as np


class QueryMeanPrecision_atk():
    def __call__(self, predicted_rank, truth_relevance,query_ids,k=3):
        # Generamos una mascara con los primeros k elementos relevantes a considerar
        predicted_rank_mask = predicted_rank < k
        # Filtramos los elementos de truth_relevance y de query_ids con los k elementos relevantes
        query_ids_k = query_ids[predicted_rank_mask]
        truth_relevance_k = truth_relevance[predicted_rank_mask]
        # Obtenemos una mascara con los elementos que son T en Truth_relevance como True
        true_relevance_mask = (truth_relevance_k == 1)
        # Filtramos los id que tienen relevance True
        filtered_query_id = query_ids_k[true_relevance_mask]
        # bincount: Cuenta la cantidad de valores filtrados iguales y almacena en la posicion equivalente al valor
        filtered_true_relevance_count = np.bincount(filtered_query_id)
        # Conservamos los valores unicos de query_ids
        unique_query_ids = np.unique(query_ids_k)
        # creamos un array con los valores filtrados por bincount distintos de cero almacenando su valor
        non_zero_count_idxs = np.where(filtered_true_relevance_count > 0)
        # crea un array de ceros del tamaño de unique_query ids + 1
        true_relevance_count = np.zeros(unique_query_ids.max() + 1)
        # obtiene un array similar al obtenido con bincount
        true_relevance_count[non_zero_count_idxs] = filtered_true_relevance_count[non_zero_count_idxs]
        # Obtiene un array como el obtenido anteriormente pero eliminando los elementos que no existen en unique_query
        true_relevance_count_by_query = true_relevance_count[unique_query_ids]
        # Obtenemos la cantidad de documentos almacenando los totales en la posicion igual al valor con bincount
        # La sintaxis con [unique_query_id] asegura de incluir solo los valores existentes en ids y eliminar los ceros
        fetched_documents_count = np.bincount(query_ids_k)[unique_query_ids]
        # Calculamos la metrica dividiendo elemento a elemento entre los documentos relevantes (true) y los totales
        precision_by_query_k = true_relevance_count_by_query / fetched_documents_count
        # Retornamos el valor medio del array anterior
        return np.mean(precision_by_query_k)
