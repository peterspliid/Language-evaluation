from evaluatemod import run
import argparse

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