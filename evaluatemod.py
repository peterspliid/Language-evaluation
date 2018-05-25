from sklearn import neighbors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import pickle
import random
import csv
import os

# So matplotlib doesn't convert text to vectors
matplotlib.rcParams['svg.fonttype'] = 'none'

# Checks if a folder exists. Creates it, if it doesn't
def check_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder

def write_report(folder, results, averages):
    report_folder = check_folder(folder + 'reports/')

    for method, method_name in {'across_areas': 'Across areas',
                                'within_areas': 'Within areas',
                                'individual_languages': 'Individual Languages'}.items():
        method_folder = check_folder(report_folder + method_name + '/')
        for feature_group_name, feature_group in results[method].items():
            if not feature_group:
                continue
            ## <feature_group_name>.txt
            if feature_group_name in averages[method]['feature_group_total']:
                text = '--Total average--\n'
                fg = averages[method]['feature_group_total'][feature_group_name]
                text += 'Score: {:1.4f}\n'.format(fg['score'])
                text += 'Base: {:1.4f}\n'.format(fg['base'])
                text += 'Count: {}\n\n'.format(fg['count'])
            if feature_group_name in averages[method]['feature_group'] and \
                'areas' in averages[method]['feature_group'][feature_group_name]:
                text += '--Area average--\n'
                for area_name, area in \
                    averages[method]['feature_group'][feature_group_name]['areas'].items():
                    text += '-{}-\n'.format(area_name)
                    text += 'Score: {:1.4f}\n'.format(area['score'])
                    text += 'Base: {:1.4f}\n'.format(area['base'])
                    text += 'Count: {}\n\n'.format(area['count'])
            for feature_name, feature in feature_group.items():
                text += "--{}--\n".format(feature_name)
                if feature:
                    text += "-Average-\n"
                    # Feature average
                    fa = averages[method]['features'][feature_group_name][feature_name]
                    text += "Score: {:1.4f}\n".format(fa['score'])
                    text += "Base: {:1.4f}\n".format(fa['base'])
                    text += "Count: {}\n\n".format(fa['count'])
                    for area_name, area in feature.items():
                        text += "-{}-\n".format(area_name)
                        if area:
                            if method != 'individual_languages':
                                text += "Amount in training: {}\n".format(area["amount_in_training"])
                            text += "Amount in test: {}\n".format(area["amount_in_test"])
                            text += "Score: {:1.4f}\n".format(area["score"])
                            text += "Base: {:1.4f}\n\n".format(area["base"])
                        else:
                            text += "No test data\n\n"
                else:
                    text += "Not enough languages\n\n"

            if feature_group_name in averages[method]['feature_group'] and \
                'languages' in averages[method]['feature_group'][feature_group_name]:
                text += "--Individual Languages--\n"
                for lang_name, lang in averages[method]['feature_group'][feature_group_name]['languages'].items():
                    text += "-{}-\n".format(lang_name)
                    text += "Score: {:1.4f}\n".format(lang['score'])
                    text += "Base: {:1.4f}\n".format(lang['base'])
                    text += "Count: {}\n\n".format(lang['count'])
            report = open(method_folder + feature_group_name + '.txt', 'w')
            report.write(text)
            report.close()

        ## Total.txt
        text = "--Total--\n"
        text += 'Score: {:1.4f}\n'.format(averages[method]['total']['score'])
        text += 'Base: {:1.4f}\n'.format(averages[method]['total']['base'])
        text += 'Count: {}\n\n'.format(averages[method]['total']['count'])
        text += '--Areas--\n'
        for area_name, area in averages[method]['area'].items():
            text += '-{}-\n'.format(area_name)
            text += 'Score: {:1.4f}\n'.format(area['score'])
            text += 'Base: {:1.4f}\n'.format(area['base'])
            text += 'Count: {}\n\n'.format(area['count'])
        text += '--Feature groups--\n'
        for feature_group_name, feature_group in averages[method]['feature_group_total'].items():
            text += '-{}-\n'.format(feature_group_name)
            text += 'Score: {:1.4f}\n'.format(feature_group['score'])
            text += 'Base: {:1.4f}\n'.format(feature_group['base'])
            text += 'Count: {}\n\n'.format(feature_group['count'])
        if method == 'individual_languages':
            text += '--Individual Languages--\n'
            for lang_name, lang in averages[method]['languages'].items():
                text += '-{}-\n'.format(lang_name)
                text += 'Score: {:1.4f}\n'.format(lang['score'])
                text += 'Base: {:1.4f}\n'.format(lang['base'])
                text += 'Number of features: {}\n\n'.format(lang['count'])
        report = open(method_folder + 'Total.txt', 'w')
        report.write(text)
        report.close()

def graph(folder, results, averages):
    graph_folder = check_folder(folder + 'graphs/bars/')

    for method, method_name in {'across_areas': 'Across areas',
                                'within_areas': 'Within areas',
                                'individual_languages': 'Individual Languages'}.items():

        # Used to make sure the areas are in the same order in all graphs
        sorted_areas = sorted(averages[method]['area'].keys())

        method_folder = check_folder(graph_folder + method_name + '/')

        # Area total
        scores = []
        bases = []
        for area_name in sorted_areas:
            area = averages[method]['area'][area_name]
            scores.append(area['score'])
            bases.append(area['base'])
        create_bar_graph(method_folder + 'Areas.png', scores, bases,
                         sorted_areas, method_name + ' - Areas')
        # Feature groups total
        scores = []
        bases = []
        labels = []
        for feature_group_name, feature_group in averages[method]['feature_group_total'].items():
            scores.append(feature_group['score'])
            bases.append(feature_group['base'])
            labels.append(feature_group_name)
        create_bar_graph(method_folder + 'Feature groups.png', scores, bases,
                         labels, method_name + ' - Feature groups')

        for feature_group_name, feature_group in results[method].items():
            if not feature_group:
                continue
            feature_group_folder = check_folder(method_folder + feature_group_name + '/')

            # All features graph
            scores = []
            bases = []
            labels = []
            for feature_name in feature_group.keys():
                if (feature_group_name in averages[method]['features'] and
                   feature_name in averages[method]['features'][feature_group_name]):
                    feature = averages[method]['features'][feature_group_name][feature_name]
                    scores.append(feature['score'])
                    bases.append(feature['base'])
                    # We only wany the feature ids, since the full names are too long
                    labels.append(feature_name.split()[0])
            if scores:
                create_bar_graph(feature_group_folder+'All features.png',
                                 scores, bases, labels, feature_group_name)

            # Areas graph
            scores = []
            bases = []
            labels = []
            if feature_group_name in averages[method]['feature_group'] and \
                'areas' in averages[method]['feature_group'][feature_group_name]:
                # Feature group
                fg = averages[method]['feature_group'][feature_group_name]['areas']
                for area_name in sorted_areas:
                    if area_name in fg:
                        scores.append(fg[area_name]['score'])
                        bases.append(fg[area_name]['base'])
                        labels.append(area_name)
                if scores:
                    create_bar_graph(feature_group_folder+'Areas.png',
                                     scores, bases, labels, feature_group_name)

            # Graphs for each individual feature
            feature_folder = check_folder(feature_group_folder + 'features/')
            for feature_name, feature in feature_group.items():
                scores = []
                bases = []
                labels = []
                for area_name in sorted_areas:
                    if area_name in feature:
                        scores.append(feature[area_name]['score'])
                        bases.append(feature[area_name]['base'])
                        labels.append(area_name)
                if scores:
                    feature_id = feature_name.split()[0]
                    create_bar_graph(feature_folder+feature_id+'.png',
                                     scores, bases, labels, feature_name)


def create_bar_graph(location, scores, bases, labels, title):
    if not scores:
        return

    font_size = 9
    rotation = 15
    # Horizontal alignment
    ha = 'right'

    if len(scores) > 40:
        font_size = 5
        rotation = 90
        ha = 'center'
    elif len(scores) > 25:
        font_size = 6
        rotation = 45
        ha = 'center'
    elif len(scores) > 15:
        font_size = 8
        rotation = 45

    indexes = np.arange(1.0, len(scores)+1)

    fig = plt.figure()
    ax = fig.add_subplot(111, title=title)
    bar_width = 0.2
    ax.bar(indexes, scores, bar_width, label="Scores")
    ax.bar(indexes + bar_width, bases, bar_width, label="Bases")
    ax.set_xticks(indexes + bar_width / 2)
    ax.set_xticklabels(labels, {'fontsize': font_size, 'ha': ha}, rotation=rotation)
    ax.legend()
    fig.savefig(location)
    plt.close(fig)

def maps(folder, averages, language_data):
    maps_folder = check_folder(folder + 'graphs/maps/')
    # Foor looking up coordinates
    lang_coords = {lang[1]: (float(lang[5]), float(lang[4])) for lang in language_data}

    # Average over all languages
    single_map(averages['individual_languages']['languages'], lang_coords,
        'Average over all languages', maps_folder)

    # Each feature group over all languages
    for fg_name, fg in averages['individual_languages']['feature_group'].items():
        if 'languages' in fg and fg['languages']:
            single_map(fg['languages'], lang_coords,
                       '{} in all languages'.format(fg_name), maps_folder)

    # Specific areas
    lang_areas = {}
    for lang in language_data:
        # lang[8] is the language area
        if not lang[8] in lang_areas:
            lang_areas[lang[8]] = {}
        if lang[1] in averages['individual_languages']['languages']:
            lang_areas[lang[8]][lang[1]] = averages['individual_languages']['languages'][lang[1]]

    for area_name, area in lang_areas.items():
        area_folder = check_folder(maps_folder + area_name + '/')
        # Average for that area
        single_map(area, lang_coords, 'Average in {}'.format(area_name), area_folder)

        # For specific feature groups
        for fg_name, fg in averages['individual_languages']['feature_group'].items():
            area_fg = {}
            if not 'languages' in fg:
                continue
            langs = fg['languages']
            for lang_name in area.keys():
                if lang_name in langs:
                    area_fg[lang_name] = langs[lang_name]
            if area_fg:
                single_map(area_fg, lang_coords, '{} in {}'.format(fg_name, area_name), area_folder)

def single_map(languages, lang_coords, title, location):
    scores = {lang_name: lang['score'] - lang['base'] for lang_name, lang in languages.items()}

    fig = plt.figure(figsize=(16, 8))
    # Since Papunesia is spread across the Pacific Ocean, we rotate the globe
    # 180 degrees
    if 'Papunesia' in title:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
        #X, Y = ccrs.transform_points(ccrs.PlateCarree(central_longitude=180), X, Y)
    else:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    fig.suptitle(title, fontsize=20)

    # Adding features to the map
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    normalize = colors.Normalize(vmin=-1, vmax=1)

    for lang_name, lang in languages.items():
        x, y = lang_coords[lang_name]
        scatter = ax.scatter(x, y, c=scores[lang_name], cmap='RdYlGn',
                             marker='o', gid=lang_name, norm=normalize,
                             transform=ccrs.PlateCarree())

    # Vertical color bar
    cbar = fig.colorbar(scatter, fraction=0.015)

    cbar.set_label('Score minus base', fontsize=15)

    plt.savefig(location + title + '.svg', bbox_inches='tight')
    plt.close(fig)

def count_score_graph(folder, averages, language_data):
    graphs_folder = check_folder(folder + 'graphs/')

    for method, method_name in {'across_areas': 'Across areas',
                                'within_areas': 'Within areas',
                                'individual_languages': 'Individual Languages'}.items():
        count_folder = check_folder(graphs_folder + 'count/' + method_name + '/')

        # All features
        fig = plt.figure()
        ax = fig.add_subplot(111, title='Amount of languages in features vs. score')
        x = []
        y = []
        for feature_group in averages[method]['features'].values():
            for feature_name, feature in feature_group.items():
                score = feature['score'] - feature['base']
                ax.scatter(feature['count'], score, gid=feature_name.split()[0])
                x.append(feature['count'])
                y.append(score)

        # Best fit line
        # https://stackoverflow.com/questions/22239691/code-for-line-of-best-fit-of-a-scatter-plot-in-python
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

        ax.set_xlabel('Amount of languages tested')
        ax.set_ylabel('Score minus base')

        plt.savefig(count_folder + 'All features.svg')
        plt.close(fig)

        # Feature groups
        for feature_group_name, feature_group in averages[method]['features'].items():
            if not feature_group:
                continue
            fig = plt.figure()
            ax = fig.add_subplot(111, title='Amount of languages in features vs. score in ' + feature_group_name)
            x = []
            y = []
            for feature_name, feature in feature_group.items():
                score = feature['score'] - feature['base']
                ax.scatter(feature['count'], score, gid=feature_name.split()[0])
                x.append(feature['count'])
                y.append(score)
            if len(x) > 1:
                ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
                ax.set_xlabel('Amount of languages tested')
                ax.set_ylabel('Score minus base')
                plt.savefig(count_folder + feature_group_name + '.svg')
            plt.close(fig)

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