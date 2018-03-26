from sklearn import neighbors
import numpy as np
import pickle
import csv

embeddings = np.load('lvec_p1.npy')

with open('lvec_ids.pkl', 'rb') as file:
    langNames = pickle.load(file)

# Used to fast determine which languages to use
langDict = dict((l.split('-')[0], l.split('-')[1][:4]) for l in langNames)

# Used to fast lookup a language embedding
embDict = dict((k[:3], v) for (k, v) in zip(langNames, embeddings))

with open('language.csv', 'rt', encoding='utf8') as file:
    reader = csv.reader(file)
    allLanguages = list(reader)

# First row is the header row
headers = allLanguages.pop(0)

# We only want languages we have embeddings for, and those generated with latn
languages = [lang for lang in allLanguages if
             lang[1] in langDict and langDict[lang[1]] == 'latn']

# List of the language families
families = sorted(list(set([lang[8] for lang in languages])))

# Feature index. The index of the first feature in the language list
fi = 10
# Minimum languages. The minimum number of languages a feature must have to
# be included
ml = 100
report = open('report.txt', 'w')
for i, feature in enumerate(headers[fi:]):
    # Languages with the selected feature
    langs = [lang for lang in languages if lang[i+fi]]
    if len(langs) < ml:
        continue

    report.write("--{}--\n".format(feature))
    for family in families:
        featureTraining = [int(lang[i+fi].split(' ')[0]) for lang in langs if lang[8] != family]
        embTraining = [embDict[lang[1]] for lang in langs if lang[8] != family]
        featureTest = [int(lang[i+fi].split(' ')[0]) for lang in langs if lang[8] == family]
        embTest = [embDict[lang[1]] for lang in langs if lang[8] == family]

        report.write("-{}-\n".format(family))
        if (featureTest):
            report.write("Number in training: {}\n".format(len(featureTraining)))
            report.write("Number in test: {}\n".format(len(featureTest)))
            knn = neighbors.KNeighborsClassifier(10, weights='uniform')
            score = knn.fit(embTraining, featureTraining).score(embTest, featureTest)
            report.write("Score: {:1.4f}\n\n" .format(score))
        else:
            report.write("No test data\n\n")
    report.write("\n")
report.close()