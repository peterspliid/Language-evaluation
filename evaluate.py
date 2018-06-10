from sklearn import neighbors
import argparse
import pickle
import csv
import os

from evaluate_module import evaluate, calculate_averages
from output import write_report, graph, maps, count_score_graph

def run(embeddings_file, report = False, graphs = False, knn_k = 10,
        classifier = 'knn', selected_feature_groups = None,
        selected_features = None, folder = 'output', echo = False):

    if not os.path.isfile(embeddings_file):
        print('Could not find embeddings file {}'.format(embeddings_file))
        return

    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)

    if folder and not folder.endswith("/"):
        folder += "/"

    if folder and not os.path.isdir(folder):
        print("Could not find the path {}".format(folder))
        return

    with open('language.csv', 'rt', encoding='utf8') as file:
        reader = csv.reader(file)
        languages = list(reader)
    headers = languages.pop(0)
    # Remove languages we do not have embeddings for
    languages = [lang for lang in languages if lang[1] in embeddings]

    with open('feature_groups.csv', 'rt', encoding='utf8') as file:
        reader = csv.reader(file)
        feature_groups = {rows[0]:rows[1] for rows in reader}

    included_features = get_included_features(feature_groups, selected_feature_groups,
                                              selected_features)
    if classifier == 'knn':
        classifier = neighbors.KNeighborsClassifier(knn_k)
    elif classifier == 'nn':
        from sklearn.neural_network import MLPClassifier
        # ‘lbfgs’ is an optimizer in the family of quasi-Newton methods, which
        # should work well with 'smaller' datasets
        classifier = MLPClassifier(solver='lbfgs')
    else:
        classifier = neighbors.KNeighborsClassifier(knn_k)

    results = evaluate(languages, headers, embeddings, included_features, classifier)
    averages = calculate_averages(results)

    if report:
        write_report(folder, results, averages)

    if graphs:
        graph(folder, results, averages)
        maps(folder, averages, languages)
        count_score_graph(folder, averages, languages)

    # Printing the total results
    if echo:
        for method, method_name in {'across_areas': 'Across areas',
                                    'within_areas': 'Within areas',
                                    'individual_languages': 'Individual Languages'}.items():
            print('{}:'.format(method_name))
            print('Score: {:1.4f}'.format(averages[method]['total']['score']))
            print('Base: {:1.4f}\n'.format(averages[method]['total']['base']))

    return (results, averages)

def get_included_features(features, selected_feature_groups, selected_features):
    included_features = 'all'
    if selected_feature_groups:
        feature_group_map = ['None', 'Phonology', 'Morphology',
                             'Nominal Categories', 'Nominal Syntax',
                             'Verbal Categories', 'Word Order',
                             'Simple Clauses', 'Complex Sentences',
                             'Lexicon', 'Sign Languages', 'Other', 'Word Order']
        selected_feature_group_names = set([feature_group_map[fg_id] for fg_id
                                        in selected_feature_groups])
        included_features = set([feature for feature, group in features.items()
                                 if group in selected_feature_group_names])
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
        choices=['knn', 'svm', 'nn'], help='Which classifier to use')
    argparser.add_argument('-g', '--feature-groups', nargs='+', type=int,
        help=("Which feature groups to include. Defaults to all. Choices are:\n"
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

    run(args.embeddings, args.report, args.graphs, args.knn_k, args.classifier,
        args.feature_groups, args.features, args.folder, True)

if __name__ == "__main__":
    main()