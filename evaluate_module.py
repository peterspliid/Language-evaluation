from sklearn import neighbors
import numpy as np
import random
import csv

# If the model predicts the most frequent label each time
def score_base(train_labels, test_labels):
    most_frequent = np.bincount(train_labels).argmax()
    return np.average([1 if y == most_frequent else 0 for y in test_labels])

# Calculates the averages for feature groups, areas and total
def calculate_averages(results):
    averages = {}

    for method in ['across_areas', 'within_areas', 'individual_languages']:
        averages[method] = {'features': {},
                            'feature_group': {},
                            'feature_group_total': {},
                            'area': {}}
        for feature_group_name, feature_group in results[method].items():
            # For calculating the average score for each language grouped by
            # feature group. Only used for individual languages
            feature_group_languages = {}

            averages[method]['features'][feature_group_name] = {}

            # Calculate the average area score. Dictionary: area -> score
            area_score = {}
            area_base = {}
            area_count = {}
            area_total_count = {}

            for feature_name, feature in feature_group.items():
                # If we do not have data for current feature, we skip it
                if not feature:
                    continue

                # Calculating the average feature score
                feature_score = 0
                feature_base = 0
                feature_count = 0
                feature_total_count = 0

                for area_name, area in feature.items():
                    feature_score += area['score']
                    feature_base += area['base']
                    feature_count += 1
                    feature_total_count += area['amount_in_test']

                    if area_name in area_score:
                        area_score[area_name] += area['score']
                        area_base[area_name] += area['base']
                        area_count[area_name] += 1
                        area_total_count[area_name] += area['amount_in_test']
                    else:
                        area_score[area_name] = area['score']
                        area_base[area_name] = area['base']
                        area_count[area_name] = 1
                        area_total_count[area_name] = area['amount_in_test']

                    if method != 'individual_languages':
                        continue

                    # For calculating the average score for each language
                    # grouped by feature group
                    for lang_name, lang in area['langs'].items():
                        if not lang_name in feature_group_languages:
                            feature_group_languages[lang_name] =\
                                {'score': 0, 'base': 0, 'count': 0}

                        feature_group_languages[lang_name]['score'] += lang['score']
                        feature_group_languages[lang_name]['base'] += lang['base']
                        feature_group_languages[lang_name]['count'] += 1

                averages[method]['features'][feature_group_name][feature_name] =\
                    {'score': feature_score / feature_count,
                     'base': feature_base / feature_count,
                     'count': feature_total_count}

            averages[method]['feature_group'][feature_group_name] = {}

            if method == 'individual_languages' and feature_group_languages:
                langs = {}
                for lang_name, lang in feature_group_languages.items():
                    count = lang['count']
                    langs[lang_name] = {'score': lang['score'] / count,
                                        'base': lang['base'] / count,
                                        'count': count}

                averages[method]['feature_group'][feature_group_name]['languages'] = langs

            # Skip if no values
            if not area_score:
                continue

            averages[method]['feature_group'][feature_group_name]['areas'] = {}

            for area_name, score in area_score.items():
                averages[method]['feature_group'][feature_group_name]['areas'][area_name] =\
                    {'score': score/area_count[area_name],
                     'base': area_base[area_name]/area_count[area_name],
                     'count': area_total_count[area_name]}

            group_score = 0
            group_base = 0
            group_count = 0
            group_total_count = 0

            for area in averages[method]['feature_group'][feature_group_name]['areas'].values():
                group_score += area['score']
                group_base += area['base']
                group_count += 1
                group_total_count += area['count']

            # Calculating the total average for each feature group
            # (averages the values in the areas)
            averages[method]['feature_group_total'][feature_group_name] =\
                {'score': group_score / group_count,
                 'base': group_base / group_count,
                 'count': group_total_count}

        averages[method]['area']
        total_area_score = {}
        total_area_base = {}
        total_area_count = {}
        total_area_total_count = {}
        for feature_group in averages[method]['feature_group'].values():
            if 'areas' in feature_group:
                for area_name, area in feature_group['areas'].items():
                    if area_name in total_area_score:
                        total_area_score[area_name] += area['score']
                        total_area_base[area_name] += area['base']
                        total_area_count[area_name] += 1
                        total_area_total_count[area_name] += area['count']
                    else:
                        total_area_score[area_name] = area['score']
                        total_area_base[area_name] = area['base']
                        total_area_count[area_name] = 1
                        total_area_total_count[area_name] = area['count']

        for area_name, score in total_area_score.items():
            if score:
                averages[method]['area'][area_name] =\
                    {'score': score / total_area_count[area_name],
                     'base': total_area_base[area_name] / total_area_count[area_name],
                     'count': total_area_total_count[area_name]}

        total_score = 0
        total_base = 0
        total_count = 0
        total_total_count = 0
        for area in averages[method]['area'].values():
            total_score += area['score']
            total_base += area['base']
            total_count += 1
            total_total_count += area['count']

        averages[method]['total'] = {'score': total_score / total_count,
                                     'base': total_base / total_count,
                                     'count': total_total_count}

    # Calculating the averages for each languages across all feature groups
    langs = {}
    for fg in averages['individual_languages']['feature_group'].values():
        if not 'languages' in fg:
            continue
        for lang_name, lang in fg['languages'].items():
            if lang_name not in langs:
                langs[lang_name] = {'score': 0, 'base': 0, 'count': 0,
                                    'total_count': 0}
            langs[lang_name]['score'] += lang['score']
            langs[lang_name]['base'] += lang['base']
            langs[lang_name]['count'] += 1
            langs[lang_name]['total_count'] += lang['count']

    averages['individual_languages']['languages'] = {}
    for lang_name, lang in langs.items():
        averages['individual_languages']['languages'][lang_name] =\
            {'score': lang['score']/lang['count'],
             'base': lang['base']/lang['count'],
             'count': lang['total_count']}

    return averages

def evaluate(languages, headers, embeddings, included_features, classifier):
    all_features = included_features == 'all'

    # Used to group the results
    with open('feature_groups.csv', 'rt', encoding='utf8') as file:
        reader = csv.reader(file)
        feature_groups = {rows[0]:rows[1] for rows in reader}
    # Unique feature groups
    u_feature_groups = list(set(feature_groups.values()))


    # Results of training and testing across language areas
    across_areas = {}
    # Results of training and testing within each language area
    within_areas = {}
    # Results of training 90% and testing 10% on all languages randomly
    individual_languages = {}

    # For grouping the feaures, which makes it easier later to average
    # and group the ouput
    for fg in u_feature_groups:
        across_areas[fg] = {}
        within_areas[fg] = {}
        individual_languages[fg] = {}

    # Unique list of language areas
    areas = sorted(list(set([lang[8] for lang in languages])))

    # Feature index. The index of the first feature in the language list
    fi = 10

    for i, feature in enumerate(headers[fi:]):
        # Skip feature if not included by user input
        if not all_features and not feature.split()[0] in included_features:
            continue

        # Feature group name
        fg = feature_groups[feature.split()[0]]

        # Initializes a dict for the language feature
        across_areas[fg][feature] = {}
        within_areas[fg][feature] = {}
        individual_languages[fg][feature] = {}

        # Languages sorted into area groups
        area_labels = {area: [] for area in areas}
        area_embeddings = {area: [] for area in areas}

        for lang in languages:
            if lang[i+fi]:
                # language[8] is the language area
                area_labels[lang[8]].append(int(lang[fi+i].split(' ')[0]))
                area_embeddings[lang[8]].append(embeddings[lang[1]])

        for area in areas:
            # If there are no languages in the current language area
            # we skip it
            if not area_labels[area]:
                continue

            ## Across areas ##
            labels_training = [label for fam in areas if fam != area for label in area_labels[fam]]
            emb_training = [embedding for fam in areas if fam != area for embedding in area_embeddings[fam]]

            labels_test = area_labels[area]
            emb_test = area_embeddings[area]

            # We skip if the training set is smaller than k
            if (not (type(classifier) == neighbors.classification.KNeighborsClassifier and
                    len(labels_training) < classifier.get_params()['n_neighbors']) and
                labels_test and labels_test and emb_training and labels_training):
                across_areas[fg][feature][area] = {}
                model = classifier.fit(emb_training, labels_training)
                score = model.score(emb_test, labels_test)
                base = score_base(labels_training, labels_test)
                across_areas[fg][feature][area]["amount_in_training"] = len(labels_training)
                across_areas[fg][feature][area]["amount_in_test"] = len(labels_test)
                across_areas[fg][feature][area]["score"] = score
                across_areas[fg][feature][area]["base"] = base

            ## Within areas ##
            # We train on 80% and test on 20%
            amount_in_training = int(round(len(area_labels[area])*0.8))

            labels_training = area_labels[area][:amount_in_training]
            emb_training = area_embeddings[area][:amount_in_training]

            labels_test = area_labels[area][amount_in_training:]
            emb_test = area_embeddings[area][amount_in_training:]

            # We skip if the training set is smaller than k
            if ((type(classifier) == neighbors.classification.KNeighborsClassifier and
                    amount_in_training < classifier.get_params()['n_neighbors']) or
                not labels_test or not labels_training or not emb_training or not labels_training):
                continue
            within_areas[fg][feature][area] = {}
            model = classifier.fit(emb_training, labels_training)
            score = model.score(emb_test, labels_test)
            base = score_base(labels_training, labels_test)
            within_areas[fg][feature][area]["amount_in_training"] = len(labels_training)
            within_areas[fg][feature][area]["amount_in_test"] = len(labels_test)
            within_areas[fg][feature][area]["score"] = score
            within_areas[fg][feature][area]["base"] = base

        ## Individual languages ##
        #import ipdb; ipdb.set_trace(context=11)
        # To find language area and feature from the language
        language_lookup = {lang[1]: [lang[8], lang[i+fi]] for lang in languages if lang[i+fi]}
        lang_ids = list(language_lookup.keys())
        # Shuffle languages, so they will not always be in alphabetical order
        random.shuffle(lang_ids)

        # We test on 10% at a time
        testing_size = int(round(len(lang_ids)*0.1))

        start = 0
        end = testing_size

        for i in range(0, 10):
            # Making sure we get the last languages, which we could have missed
            # due to rounding
            if i == 9:
                training_languages = lang_ids[:start]
                testing_languages = lang_ids[start:]
            else:
                training_languages = lang_ids[:start] + lang_ids[end:]
                testing_languages = lang_ids[start:end]

            labels_training = [int(language_lookup[lang][1].split()[0]) for lang in training_languages]
            emb_training = [embeddings[lang] for lang in training_languages]

            labels_test = [int(language_lookup[lang][1].split()[0]) for lang in testing_languages]
            emb_test = [embeddings[lang] for lang in testing_languages]

            # We skip if the training set is smaller than k
            if not (type(classifier) == neighbors.classification.KNeighborsClassifier and
                    len(labels_training) < classifier.get_params()['n_neighbors']):

                if not (emb_test and emb_training):
                    continue
                model = classifier.fit(emb_training, labels_training)

                predictions = model.predict(emb_test)
                most_frequent = np.bincount(labels_training).argmax()

                for lang, label, prediction in zip(testing_languages, labels_test, predictions):
                    area = language_lookup[lang][0]
                    if not area in individual_languages[fg][feature]:
                        # We are keeping the same structures as the other
                        # methods
                        individual_languages[fg][feature][area] = {}
                        individual_languages[fg][feature][area]['langs'] = {}

                    individual_languages[fg][feature][area]['langs'][lang] =\
                        {'score': 1 if label == prediction else 0,
                         'base': 1 if most_frequent == label else 0}

            start += testing_size
            end += testing_size

        # To keep the same structure as the other methods, we calculate
        # the area averages here
        for area_name, area in individual_languages[fg][feature].items():
            score = 0
            base = 0
            count = 0
            for lang in area['langs'].values():
                score += lang['score']
                base += lang['base']
                count += 1
            individual_languages[fg][feature][area_name]['score'] = score/count
            individual_languages[fg][feature][area_name]['base'] = base/count
            individual_languages[fg][feature][area_name]['amount_in_test'] = count

    results = {'across_areas': across_areas,
               'within_areas': within_areas,
               'individual_languages': individual_languages}
    return results