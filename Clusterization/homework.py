import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk_cluster
import sklearn.preprocessing as sk_preproc
import scipy.cluster.hierarchy as sci_clust
import numpy as np

#agglomerative кластеризация
def set_print_opt():
    pd.set_option('display.max_columns',None)
    pd.set_option('display.width',None)
set_print_opt()


clients = pd.read_csv('data\customer_online_closing_store.csv')
print(clients)

#нужно исследовать корреляцию возвратов и покупок с рейтингом и попробовать модель от єтого

clients['return_rate'] = clients['items_returned']/clients['items_purchased']# how often people return
clients['average_price'] = clients['total_spent']/clients['items_purchased']

#нюанс - обшие затраты при частом возврате не говорит о покупателе много

X = clients[['average_price','return_rate','overall_rating']]
print(X)

min_max_scaler = sk_preproc.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
print(X)

linkage_method = 'ward'
dendrogram = sci_clust.dendrogram(sci_clust.linkage(X,method=linkage_method))
aggl_model = sk_cluster.Birch()
aggl_model.fit(X)

clients['class'] = aggl_model.labels_
print(clients[['average_price','return_rate','overall_rating','class']])

client_pivot_table = clients.pivot_table(index='class',
                                         values=['average_price','return_rate','overall_rating','customer_id'],
                                         aggfunc={'average_price':np.mean, 'return_rate':np.mean,
                                                  'overall_rating':np.mean, 'customer_id':len
                                                  })
print(client_pivot_table)
plt.show()