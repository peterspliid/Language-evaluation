from sklearn import neighbors, svm
import argparse
import pickle
import csv
import os
import numpy as np

from evaluate_module import evaluate, calculate_averages
from output import write_report, graph, maps, count_score_graph

def run_evaluation(embeddings, report = False, graphs = False, knn_k = 10,
        classifier = 'knn', selected_feature_areas = None,
        selected_features = None, folder = 'output'):

    if not verify_embeddings(embeddings):
        return (False, "Wrong embeddings formart. Format must be a dictionary where the keys are language IDs (ISO 639-3) and values are the language embeddings")

    if folder and not folder.endswith("/"):
        folder += "/"

    if folder and not os.path.isdir(folder):
        return (False, "Could not find the path {}".format(folder))

    print("Starting")

    with open('language.csv', 'rt', encoding='utf8') as file:
        reader = csv.reader(file)
        languages = list(reader)
    headers = languages.pop(0)
    # Remove languages we do not have embeddings for
    languages = [lang for lang in languages if lang[1] in embeddings]

    with open('feature_areas.csv', 'rt', encoding='utf8') as file:
        reader = csv.reader(file)
        feature_areas = {rows[0]:rows[1] for rows in reader}

    included_features = get_included_features(feature_areas, selected_feature_areas,
                                              selected_features)
    if classifier == 'knn':
        classifier = neighbors.KNeighborsClassifier(knn_k)
    elif classifier == 'mlp':
        from sklearn.neural_network import MLPClassifier
        # ‘lbfgs’ is an optimizer in the family of quasi-Newton methods, which
        # should work well with 'smaller' datasets
        classifier = MLPClassifier(solver='lbfgs')
    else:
        classifier = svm.SVC()

    print("Evaluating embeddings")
    results = evaluate(languages, headers, embeddings, included_features, classifier)
    print("Calculating averages")
    averages = calculate_averages(results)

    if report:
        print("Writing text reports")
        write_report(folder, results, averages)

    if graphs:
        print("Creating bar graphs")
        graph(folder, results, averages)
        print("Creating maps")
        maps(folder, averages, languages)
        print("Creating count graphs")
        count_score_graph(folder, averages, languages)

    print("Finished\n")

    return (True, (results, averages))

def verify_embeddings(embeddings):
    if type(embeddings) != dict:
        return False

    for lang, emb in embeddings.items():
        if (type(lang) != str or len(lang) != 3 or
         type(emb) not in [list, np.ndarray] or len(emb) == 0):
            return False
    return True

def get_included_features(features, selected_feature_areas, selected_features):
    included_features = 'all'
    if selected_feature_areas:
        feature_area_map = ['None', 'Phonology', 'Morphology',
                             'Nominal Categories', 'Nominal Syntax',
                             'Verbal Categories', 'Word Order',
                             'Simple Clauses', 'Complex Sentences',
                             'Lexicon', 'Sign Languages', 'Other', 'Word Order']
        selected_feature_area_names = set([feature_area_map[fg_id] for fg_id
                                        in selected_feature_areas])
        included_features = set([feature for feature, area in features.items()
                                 if area in selected_feature_area_names])
        if selected_features:
            included_features |= set([f.upper() for f in selected_features])
    return included_features
def main():
    argparser = argparse.ArgumentParser(description="Evaluate language representations",
                                        formatter_class=argparse.RawTextHelpFormatter)

    # Command line arguments
    argparser.add_argument('embeddings', help="A pickle file, which consist of a dictionary where the keys are language IDs (ISO 639-3) and values are the language embeddings")
    argparser.add_argument('-r', '--report', action='store_true',
        help="Write text reports with all the results")
    argparser.add_argument('-d', '--graphs', action='store_true',
        help="Create graphs with all the results (takes a couple of minutes)")
    argparser.add_argument('-k', '--knn-k', default=17, type=int,
        help="K value for the knn classifier")
    argparser.add_argument('-c', '--classifier', default='knn',
        choices=['knn', 'svm', 'mlp'], help='Which classifier to use')
    argparser.add_argument('-g', '--feature-areas', nargs='+', type=int,
        help=("Which feature areas to include. Defaults to all. Choices are:\n"
              "0   - None (add individual features with -f)\n"
              "1   - Phonology\n"
              "2   - Morphology\n"
              "3   - Nominal Categories\n"
              "4   - Nominal Syntax\n"
              "5   - Verbal Categories\n"
              "6   - Word Order\n"
              "7   - Simple Clauses\n"
              "8   - Complex Sentences\n"
              "9   - Lexicon\n"
              "10  - Sign Languages\n"
              "11  - Other\n"
              "12  - Word Order"))
    argparser.add_argument('-f', '--features', nargs='+',
        help="Which features to include. Only choose features not included by -g. Use IDs from wals.info")
    argparser.add_argument('-o', '--output-folder', dest='folder', default='output',
        help='Where to save the output files')

    args = argparser.parse_args()

    if not os.path.isfile(args.embeddings):
        print('Could not find embeddings file {}'.format(args.embeddings))

    with open(args.embeddings, 'rb') as f:
        embeddings = pickle.load(f)

    success, data = run_evaluation(embeddings, args.report, args.graphs, args.knn_k,
        args.classifier, args.feature_areas, args.features, args.folder)

    # On success print the total averages, other print the error
    if success:
        averages = data[1]
        for method, method_name in {'across_areas': 'Across areas',
                                    'within_areas': 'Within areas',
                                    'individual_languages': 'Individual Languages'}.items():
            print('{}:'.format(method_name))
            print('Score: {:1.4f}'.format(averages[method]['total']['score']))
            print('Base: {:1.4f}\n'.format(averages[method]['total']['base']))
    else:
        print(data)

if __name__ == "__main__":
    main()