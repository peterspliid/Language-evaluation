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
            average_score = 0
            average_base = 0
            amount_in_average = 0
            for family_name, family in feature.items():
                report.write("-{}-\n".format(family_name))
                if family:
                    report.write("Number in training: {}\n".format(family["numberInTraining"]))
                    report.write("Number in test: {}\n".format(family["numberInTest"]))
                    report.write("Score: {:1.4f}\n".format(family["score"]))
                    report.write("Base: {:1.4f}\n\n".format(family["base"]))
                    average_score += family["score"]
                    average_base += family["base"]
                    amount_in_average += 1
                else:
                    report.write("No test data\n\n")
            report.write("-Average-\n")
            report.write("Score: {:1.4f}\n".format(average_score/amount_in_average))
            report.write("Base: {:1.4f}\n\n".format(average_base/amount_in_average))
        else:
            report.write("Not enough languages\n\n")
    report.close()

'''
def graph(results):
    indexes = np.arange(len(fids))
    bar_width = 0.2
    plt.bar(indexes, scores, bar_width, label="Scores")
    plt.bar(indexes + bar_width, bases, bar_width, label="Bases")
    plt.xticks(indexes + bar_width / 2, fids)
    plt.legend()
    plt.show()
'''


def total_average_scores(results):
    average_score = 0
    average_base = 0
    amount_in_average = 0
    for feature in results.values():
        if feature:
            for family in feature.values():
                if family:
                    average_score += family["score"]
                    average_base += family["base"]
                    amount_in_average += 1
    return (average_score/amount_in_average, average_base/amount_in_average)

def score_normal(model, features, labels):
    predictions = model.predict(features)
    return np.average([1 if x == y else 0 for x, y in zip(predictions, labels)])

# If the model predicts the most frequent label each time
def score_base(labels):
    mostFrequent = np.bincount(labels).argmax()
    return np.average([1 if y == mostFrequent else 0 for y in labels])

def evaluate(emb_dict, languages, headers, knn_k = 1):
    # List of the language families
    families = sorted(list(set([lang[8] for lang in languages])))

    # Feature index. The index of the first feature in the language list
    fi = 10

    # Minimum languages. The minimum number of languages a feature must have to
    # be included
    ml = 100

    #Feature IDs from WALS
    results = {}

    for i, feature in enumerate(headers[fi:]):
        # Initializes a dict for the language feature
        results[feature] = {}

        # Languages with the selected feature
        langs = [lang for lang in languages if lang[i+fi]]

        # Skips is feature doesn't have required amount of languages
        if len(langs) < ml:
            continue

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
    return results

def main():
    ## Parameters ##

    # Write a text report with the results
    txt_report = True

    # Generate graphs with the resulted data
    graph = False

    # Test a number of different ks in the KNN classifier and output a graph
    test_k = False

    # Output folder. Where to output graphs and results
    r_folder = "output\\"

    embeddings = np.load('lvec_p1.npy')

    #import ipdb; ipdb.set_trace()

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

    if test_k:
        max_k = 30
        scores = []
        for k in range(1, max_k):
            r = evaluate(emb_dict, languages, headers, k)
            av_score, _ = total_average_scores(r)
            scores.append(av_score)
        print(scores)
        '''
        k_fig = plt.figure()
        k_ax = k_fig.add_subplot(111)
        k_ax.plot(range(1, max_k), scores)
        k_ax.xticks(range(1,max_k))
        k_fig.savefig(r_folder + "k_fig.pdf")
        '''

    results = evaluate(emb_dict, languages, headers)

    if txt_report:
        write_report("as", results)

#    if graph:
 #       graph(results)

    av_score, av_base = total_average_scores(results)
    print(av_score)
    print(av_base)

if __name__ == "__main__":
    main()