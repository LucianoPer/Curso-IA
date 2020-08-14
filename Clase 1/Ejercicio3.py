import numpy as np


class indexer(object):
    def __init__(self, ids):
        # genera un array con valores unicos y ordenados
        user_ids = np.unique(ids)
        # genera un array de -1 con la cantidad del valor max del array ids + 1
        id_idx = np.ones(user_ids.max() + 1, dtype=np.int64) * -1
        # cargamos el array id_idx en las posiciones iguales al valor de ids con valores decorridos ( indice )
        id_idx[user_ids] = np.arange(user_ids.size)
        self.id_idx = id_idx
        self.idx_id = user_ids

    def get_idx(self, ids):
        ids = self.id_idx[ids]
        return ids, ids != -1

    def get_id(self, idx):
        return self.idx_id[idx]
