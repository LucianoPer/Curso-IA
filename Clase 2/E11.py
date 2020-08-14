import numpy as np


def dataset_generator(dim, n_element, amplitud):
    dataset = amplitud * np.random.rand(n_element,dim)
    dataset = np.random.permutation(dataset)
    dataset_20 = dataset[:int(n_element*0.2),]
    dataset_10 = dataset[int(n_element*0.2):int(n_element*0.2)+int(n_element*0.1),]
    dataset_70 = dataset[int(n_element*0.2)+int(n_element*0.1):,]
    return dataset_70,dataset_20,dataset_10



# train,validation,test= dataset_generator(2,100,10)
# print("train {} //  validation {}  //  test {}  ".format(train,validation,test))
# print("train size : {}   val size {}  test size {}".format(train.size/2,validation.size/2,test.size/2))
