import numpy as np
import pickle
import csv
from sklearn import neighbors

embeddings = np.load('lvec_p1.npy')

with open('lvec_ids.pkl', 'rb') as file:
    langNames = pickle.load(file)

with open('languages.csv', 'rb') as file:
    reader = csv.DictReader(file)
    languages = {rows['iso_codes']: {'id': rows['id'],
                              'macroarea': rows['macroarea']}
                             for rows in reader}

genders = {'145': 0,
           '146': 2,
           '147': 3,
           '148': 4,
           '149': 5}
with open('features/30A.csv', 'rb') as file:
    reader = csv.DictReader(file)
    f30a = {rows['id'][-3:]: genders[rows['domainelement_pk']] for rows in reader}

X = []
y = []
for i, embedding in enumerate(embeddings):
    langIso = langNames[i][:3]
    if langIso in languages:
        langId = languages[langIso]['id']
        if langId in f30a:
            X.append(embedding)
            y.append(f30a[langId])
            #print (f30a[langId], embedding)

knn = neighbors.KNeighborsRegressor(5, weights='uniform')
y_ = knn.fit(X, y).score(X, y)
print y_

#print(str(len(data))+'\n')
#print(str(len(embeddings))+'\n')