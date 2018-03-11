import numpy as np
import pickle
import csv

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

for i, lang in enumerate(embeddings):
    langIso = langNames[i][:3]
    if langIso in languages:
        print languages[langIso]

'''

for langId in f30a:
    print langId
    iso = languages[langId]
    print iso
'''
#print(str(len(data))+'\n')
#print(str(len(embeddings))+'\n')