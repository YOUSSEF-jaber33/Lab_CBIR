import numpy as np
from distances import manhattan, euclidean, chebyshev, canberra
 
def calculate_similarity(features_db, query_features, distance, num_results):
    distances = []
   
    query_features = np.array(query_features, dtype=np.float64)
   
    for instance in features_db:
        features, label, img_path = instance[:-2], instance[-2], instance[-1]
       
       
        features = np.array(features, dtype=np.float64)
       
        if distance == 'Manhattan':
            dist = manhattan(query_features, features)
        elif distance == 'Euclidean':
            dist = euclidean(query_features, features)
        elif distance == 'Chebyshev':
            dist = chebyshev(query_features, features)
        elif distance == 'Canberra':
            dist = canberra(query_features, features)
       
        distances.append((img_path, dist, label))
   
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:num_results]]
 
