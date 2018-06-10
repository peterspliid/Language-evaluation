import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
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