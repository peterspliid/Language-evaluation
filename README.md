# Evaluation of language representations
This tool can be used to evaluate language representations. The tool needs the input to be a dictionary, where the keys are language IDs and the values are the representations. There are 3 ways to run the tool
## Webservice
This is the recommended way to run the evaluation, as it gives a more complete overview of the results. To start the webservice, run the command
```
python webservice.py
```
This starts a webservice on port 80. Then go to http://localhost/ to access the webservice. To change the port, edit the file webservice.py. The webservice needs a pickle with the dictionary containing representations.
## Command line
To run the tool using the command line, the representations are loaded from a pickle file. For more information about command line, run the command
```
python evaluate.py -h
```
## Python module
To run this from a python module, the run_evaluation function can be imported from evalute.py. It has the following arguments:
```
embeddings : dictionary (string: list (int/float))
        A dictionary with the embeddings to evaluate. The keys must be the
        language identifier, and the values are the language embeddings
    report : bool
        Whether to write text reports with the results and averages in the
        output folder
    graphs : bool
        Whether to create graphs of the results in the output folder. This
        includes bar graphs, maps, and count graphs. This can take several
        minutes to complete
    classifier_args : list (int)
        A list of arguments to the classifier. If using k-nearest neighbors,
        it's k, if using multilayer perceprton, it's the layer sizes, where
        the length of the list indicates the number of layers
    classifier : string
        Which classifier to use. Possible values are knn for k-nearest neighbors,
        mlp for multilayer perceptron, and svm for support vector machine.
    selected_feature_areas : list (int)
        List of feature areas to evalute for. Leave out for all, or use ant of the
        following integers
            0   - None (add individual features with selected features)
            1   - Phonology
            2   - Morphology
            3   - Nominal Categories
            4   - Nominal Syntax
            5   - Verbal Categories
            6   - Word Order
            7   - Simple Clauses
            8   - Complex Sentences
            9   - Lexicon
            10  - Sign Languages
            11  - Other
            12  - Word Order
    selected_features : list (string)
        Add individual features not included by selected_feature_areas
    folder : string
        The output folder where to place the text reports and graphs.
```
It returns either of the following
```
    Tuple (True, (dictionary, dictionary)) or
    Tuple (False, string)
        If success, it results a tuple where the first value is true, and the
        second value is a tuple with a the results and the averages.
        If failes, the first value is false, and the second value is an error
        message.
```