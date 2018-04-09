from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

def write_report(folder, results):
    report = open('report.txt', 'w')
    for feature_name, feature in results.items():
        report.write("--{}--\n".format(feature_name))
        if feature:
            for family_name, family in feature.items():
                report.write("-{}-\n".format(family_name))
                if (family):
                    report.write("Number in training: {}\n".format(family["numberInTraining"]))
                    report.write("Number in test: {}\n".format(family["numberInTest"]))
                    report.write("Score: {:1.4f}\n".format(family["score"]))
                    report.write("Base: {:1.4f}\n\n".format(family["base"]))
                else:
                    report.write("No test data\n\n")
        else:
            report.write("Not enough languages\n\n")
    report.close()

def total_average(results, field):
    av = 0
    avn = 0
    for feature in results.values():
        if feature:
            av += feature["Average"][field]
            avn += 1
    return av/avn

def score_normal(model, features, labels):
    predictions = model.predict(features)
    return np.average([1 if x == y else 0 for x, y in zip(predictions, labels)])

# If the model predicts the most frequent label each time
def score_base(labels):
    mostFrequent = np.bincount(labels).argmax()
    return np.average([1 if y == mostFrequent else 0 for y in labels])

## Parameters ##
# k for K Neighbors Classifier
knn_k = 10

# Write a text report with the results
txt_report = False

# Minimum languages. The minimum number of languages a feature must have to
# be included
ml = 100

# Results folder. Where the graph and report is saved
r_folder = "results"

embeddings = np.load('lvec_p1.npy')

with open('lvec_ids.pkl', 'rb') as file:
    langNames = pickle.load(file)

# Used to fast determine which languages to use
lang_dict = dict((l.split('-')[0], l.split('-')[1][:4]) for l in langNames)

# Used to fast lookup a language embedding
emb_dict = dict((k[:3], v) for (k, v) in zip(langNames, embeddings))

with open('language.csv', 'rt', encoding='utf8') as file:
    reader = csv.reader(file)
    all_languages = list(reader)

# First row is the header row
headers = all_languages.pop(0)

# We only want languages we have embeddings for, and those generated with latn
languages = [lang for lang in all_languages if
             lang[1] in lang_dict and lang_dict[lang[1]] == 'latn']

# List of the language families
families = sorted(list(set([lang[8] for lang in languages])))

# Feature index. The index of the first feature in the language list
fi = 10

#Feature IDs from WALS
fids = []
scores = []
bases = []
results = {}

#report = open(f_name, 'w')
for i, feature in enumerate(headers[fi:]):
    # Initializes a dict for the language feature
    results[feature] = {}

    # Languages with the selected feature
    langs = [lang for lang in languages if lang[i+fi]]

    # Skips is feature doesn't have required amount of languages
    if len(langs) < ml:
        continue

    fids.append(feature.split(" ")[0])

    average_score = 0
    average_base = 0
    amount_in_average = 0
    for family in families:
        results[feature][family] = {}
        feature_training = [int(lang[i+fi].split(' ')[0]) for lang in langs if lang[8] != family]
        emb_training = [emb_dict[lang[1]] for lang in langs if lang[8] != family]
        feature_test = [int(lang[i+fi].split(' ')[0]) for lang in langs if lang[8] == family]
        emb_test = [emb_dict[lang[1]] for lang in langs if lang[8] == family]

        if (feature_test):
            knn = neighbors.KNeighborsClassifier(knn_k, weights='distance')
            model = knn.fit(emb_training, feature_training)
            score = model.score(emb_test, feature_test)
            base = score_base(feature_test)
            results[feature][family]["numberInTraining"] = len(feature_training)
            results[feature][family]["numberInTest"] = len(feature_test)
            results[feature][family]["score"] = score
            results[feature][family]["base"] = base

            amount_in_average += 1
            average_score += score
            average_base += base
    average_score /= amount_in_average
    average_base /= amount_in_average
    scores.append(average_score)
    bases.append(average_base)

    results[feature]["Average"] = {}
    results[feature]["Average"]["numberInTraining"] = 0
    results[feature]["Average"]["numberInTest"] = 0
    results[feature]["Average"]["score"] = average_score
    results[feature]["Average"]["base"] = average_base

if (txt_report):
    write_report("as", results)

# Plotting results to a bar graph
indexes = np.arange(len(fids))
bar_width = 0.2
plt.bar(indexes, scores, bar_width, label="Scores")
plt.bar(indexes + bar_width, bases, bar_width, label="Bases")
plt.xticks(indexes + bar_width / 2, fids)
plt.legend()
#plt.show()

print(total_average(results, "score"))
print(total_average(results, "base"))
