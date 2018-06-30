import pickle
import matplotlib.pyplot as plt
import csv
import random as r

from evaluate import run_evaluation

def test_k_graph():
    with open('featurevectors.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    scores = []
    for i in range(1, 51):
        print(i)
        success, data = run_evaluation(embeddings, selected_feature_areas=[2, 3, 4, 5, 6],
                          classifier_args=[i])
        if not success:
            print(data)
            return
        _, averages = data
        score = 0
        for model in ['across_areas', 'within_areas', 'individual_languages']:
            score += averages[model]['total']['score']
        score /= 3
        scores.append(score)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 51), scores)
    ax.set_xlabel('k')
    ax.set_ylabel('Score')
    plt.show()

def test_mlp_graph():
    with open('featurevectors.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    scores = [[], [], []]
    for i in range(0, 3):
        for size in range(10, 101, 10):
            arg = [size for _ in range(i+1)]
            print(arg)
            success, data = run_evaluation(embeddings, selected_feature_areas=[2, 3, 4, 5, 6],
                                classifier_args=arg, classifier='mlp')
            if not success:
                print(data)
                return
            _, averages = data
            score = 0
            for model in ['across_areas', 'within_areas', 'individual_languages']:
                score += averages[model]['total']['score']
            score /= 3
            scores[i].append(score)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(10, 101, 10), scores[0], label='1 layer')
    ax.plot(range(10, 101, 10), scores[1], label='2 layers')
    ax.plot(range(10, 101, 10), scores[2], label='3 layers')
    ax.set_xticklabels(range(10, 101, 10))
    ax.legend()
    ax.set_xlabel('Size of layers')
    ax.set_ylabel('Score')
    plt.show()


def test_classifier():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    scores = []
    classifiers = ['knn', 'svm', 'mlp']
    for classifier in classifiers:
        success, data = run_evaluation(embeddings, selected_feature_areas=[2, 3, 4, 5, 6],
                          classifier=classifier)
        if not success:
            print(data)
            return
        _, averages = data
        score = 0
        for model in ['across_areas', 'within_areas', 'individual_languages']:
            score += averages[model]['total']['score']
        score /= 3
        scores.append(score)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(1, 4), scores)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels(classifiers)
    ax.set_xlabel('Classifier')
    ax.set_ylabel('Score')
    plt.show()

def create_feature_vectors():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open('language.csv', 'rt', encoding='utf8') as file:
        reader = csv.reader(file)
        languages = list(reader)
    headers = languages.pop(0)
    feature_embeddings = {}
    for language in languages:
        # We test the same languages we have embeddings for, so it is easier
        # to compare
        if language[1] and language[1] in embeddings.keys():
            embedding = [int(val.split()[0]) if val else 0 for val in language[10:]]
            feature_embeddings[language[1]] = embedding
    with open('featurevectors.pkl', 'wb') as f:
        pickle.dump(feature_embeddings, f, pickle.HIGHEST_PROTOCOL)

def run_tests():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    langs = list(embeddings.keys())

    test_num = 1

    while True:
        print('Test number {}'.format(test_num))
        test_num += 1
        r.shuffle(langs)
        number_of_langs = r.randrange(0, len(embeddings))
        random_embeddings = {}
        for i, lang in enumerate(langs):
            random_embeddings[lang] = embeddings[lang]
            if i > number_of_langs:
                break

        feature_area_chance = r.random()
        feature_areas = [val for val in range(0, 13) if r.random() > feature_area_chance]

        classifier = ['knn', 'svm', 'mlp'][r.randrange(0, 3)]
        if classifier == 'knn':
          classifier_arg = [r.randrange(1, 50)]
        else:
          classifier_arg = [r.randrange(1, 200) for _ in range(r.randrange(1, 10))]
        try:
            run_evaluation(random_embeddings, True, True, classifier_arg, classifier, feature_areas, None)
        except Exception as e:
            print(e)
            print('Embeddings:')
            print(random_embeddings)
            print('')
            print('Feature areas:')
            print(feature_areas)
            print('Classifier: {}'.format(classifier))
            print('Classifier arg: {}'.format(classifier_arg))
            return

def test():
    embeddings = {}
    with open('embeddings-test.pkl', 'wb') as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
    run_evaluation(embeddings, True, True, 7, 'knn', [3])

test_classifier()