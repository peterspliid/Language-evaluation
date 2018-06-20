import pickle
import csv
import os
from io import StringIO

class EvaluationPage:
    models = {'across_areas': 'Across areas',
               'within_areas': 'Within areas',
               'individual_languages': 'Individual Languages'}

    status = 200

    def __init__(self, path):
        self.content = StringIO()
        self.session = path.pop(0)
        self.path = path

        session_dir = 'temp/' + self.session + '/'

        with open(session_dir + 'data.pkl', 'rb') as f:
            self.success, self.data = pickle.load(f)

        if self.success:
            self.results, self.averages = self.data
        else:
            self.results, self.averages = ({}, {})

    def create_page(self):
        # If an error happened during the evaluation
        if not self.success:
            self.content.write(self.data)
            return

        self.content.write(self.create_menu())
        self.content.write('<div id="evaluation-page">')

        # If front evaluation page
        if len(self.path) == 0:
            self.content.write(self.summary_page())
        elif len(self.path) == 1:
            page = self.path[0]
            if page == 'languages':
                self.content.write(self.languages_page())
            elif page == 'feature-area':
                self.content.write(self.feature_area_average_page())
            elif page == 'feature':
                self.content.write(self.all_features_page())
            elif page == 'macroarea':
                self.content.write(self.macroarea_average_page())
            else:
                self.content.write(self.unknown_page())
                self.status = 404

        elif len(self.path) == 2:
            page = self.path[0]
            if page == 'feature-area':
                self.content.write(self.individual_feature_area_page())
            elif page == 'feature':
                self.content.write(self.individual_feature_page())
            elif page == 'macroarea':
                self.content.write(self.individual_macroarea_page())
            else:
                self.content.write(self.unknown_page())
                self.status = 404
        else:
            self.content .write(self.unknown_page())
            self.status = 404

        self.content.write('</div>')

    def summary_page(self):
        self.prepare_lang_data(True, True)
        content = StringIO()
        content.write('<h1>Summary</h1>')
        content.write('<h2>Average scores</h2>')
        content.write('<div class="columns section">')
        for model, model_name in self.models.items():
            score = self.averages[model]['total']['score']
            base = self.averages[model]['total']['base']
            count = self.averages[model]['total']['count']
            content.write('<div><table class="summary"><tr><th colspan="2">{}</th></tr>'.format(model_name))
            content.write('<tr><td>Score</td><td>{:1.4f}</td></tr>'.format(score))
            content.write('<tr><td>Base</td><td>{:1.4f}</td></tr>'.format(base))
            content.write('<tr><td>Score - base</td><td>{:1.4f}</td></tr>'.format(score - base))
            content.write('<tr><td>Number of scores</td><td>{}</td></tr></table></div>'.format(count))
        content.write('</div>')
        content.write('<h2>Map of all languages</h2><div class="section">')
        text = '''
        This map shows the scores of each language as a colored dot on the map.
        The greener the dot, the higher the higher is the difference between
        the base value and the score. Hover over a colored dot to see the name
        of the language. See more under <a href="languages">languages</a>.
        '''
        content.write(self.get_graph('maps/Average over all languages.svg', '', text))

        content.write('</div><h2>Count graphs</h2>')
        content.write('''
        <div class="section">
        <p>The following graphs have a dot for each feature. They show the score
        along with the number of languages tested for each feature. This gives an
        idea whether more data makes the predictions more precise. Click on a feature
        dot to get more information about a specific feature.</p>
        <div class="inline">
        ''')
        for model_name in self.models.values():
            content.write(self.get_graph('count/{}/All features.svg'.format(model_name), model_name))
        content.write('</div></div>')
        content.write(self.javascript_tooltips(True, True))

        return content.getvalue()

    def languages_page(self):
        self.prepare_lang_data(True, False)
        content = StringIO()
        content.write('<h1>All languages</h1>')
        content.write('<h2>Language list</h2><div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Language ID</th><th>Language name</th><th>Macroarea</th><th>Score</th><th>Base</th><th>Score - base</th><th>Features tested</th></tr></thead><tbody>')
        for lang_id, language in self.averages['individual_languages']['languages'].items():
            langname, macroarea = self.langnames[lang_id]
            content.write('<tr>')
            content.write('<td>{}</td>'.format(lang_id))
            content.write('<td>{}</td>'.format(langname))
            content.write('<td>{}</td>'.format(macroarea))
            content.write('<td class="right">{:1.4f}</td>'.format(language['score']))
            content.write('<td class="right">{:1.4f}</td>'.format(language['base']))
            content.write('<td class="right">{:1.4f}</td>'.format(language['score'] - language['base']))
            content.write('<td class="right">{}</td>'.format(language['count']))
            content.write('</tr>')
        content.write('</tbody></table></div>')
        return content.getvalue()

    def feature_area_average_page(self):
        content = StringIO()
        content.write('<h1>All feature areas</h1>')
        content.write('<h2>Feature area list</h2>')
        content.write('<div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Model</th><th>Feature area</th><th>Score</th><th>Base</th><th>Score - base</th><th>Count</th></tr></thead><tbody>')
        for model, model_name in self.models.items():
            for feature_area_name, feature_area in self.averages[model]['feature_area_total'].items():
                content.write('<tr>')
                content.write('<td>{}</td>'.format(model_name))
                content.write('<td>{}</td>'.format(feature_area_name))
                content.write('<td>{:1.4f}</td>'.format(feature_area['score']))
                content.write('<td>{:1.4f}</td>'.format(feature_area['base']))
                content.write('<td>{:1.4f}</td>'.format(feature_area['score']-feature_area['base']))
                content.write('<td>{}</td>'.format(feature_area['count']))
                content.write('</tr>')
        content.write('</tbody></table>')
        content.write('</div>')

        #for model_name in self.models.values():
        #    content.write(self.get_graph('temp/{}/graphs/bars/{}/Areas.svg'.format(session, model_name)))
        content.write('<h2>Feature area graphs</h2><div class="section inline">')
        for model_name in self.models.values():
            content.write(self.get_graph('bars/{}/Feature areas.svg'.format(model_name), model_name))
        content.write('</div>')
        #for model_name in self.models.values():
        #    content.write(self.get_graph('temp/{}/graphs/count/{}/All features.svg'.format(session, model_name)))

        return content.getvalue()

    def individual_feature_area_page(self):
        feature_area_name =  self.capitalize_first_letters(self.path[1])
        print(feature_area_name)
        if not feature_area_name in self.averages['individual_languages']['feature_area'].keys():
            return self.unknown_page()


        content = StringIO()
        content.write('<h1>{}</h1>'.format(feature_area_name))
        content.write('<h2>Summary</h2>')
        content.write('<div class="columns section">')
        for model, model_name in self.models.items():
            if feature_area_name in self.averages[model]['feature_area_total']:
                feature_area_total = self.averages[model]['feature_area_total'][feature_area_name]
                content.write('<table class="summary"><tr><th colspan="2">{}</th></tr>'.format(model_name))
                content.write('<tr><td>Score</td><td>{:1.4f}</td></tr>'.format(feature_area_total['score']))
                content.write('<tr><td>Base</td><td>{:1.4f}</td></tr>'.format(feature_area_total['base']))
                content.write('<tr><td>Score - base</td><td>{:1.4f}</td></tr>'.format(feature_area_total['score']-feature_area_total['base']))
                content.write('<tr><td>Count</td><td>{}</td></tr></table>'.format(feature_area_total['count']))
        content.write('</div>')

        content.write('<h2>Macroarea list</h2>')
        content.write('<div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Model</th><th>Macroarea</th><th>Score</th><th>Base</th><th>Score - base</th><th>Languages tested</th></tr></thead><tbody>')
        for model, model_name in self.models.items():
            if 'areas' in self.averages[model]['feature_area'][feature_area_name]:
                for area_name, area in self.averages[model]['feature_area'][feature_area_name]['areas'].items():
                    content.write('<tr>')
                    content.write('<td>{}</td>'.format(model_name))
                    content.write('<td>{}</td>'.format(area_name))
                    content.write('<td class="right">{:1.4f}</td>'.format(area['score']))
                    content.write('<td class="right">{:1.4f}</td>'.format(area['base']))
                    content.write('<td class="right">{:1.4f}</td>'.format(area['score']-area['base']))
                    content.write('<td class="right">{}</td>'.format(area['count']))
                    content.write('</tr>')
        content.write('</tbody></table>')
        content.write('</div>')

        content.write('<h2>Macroarea graphs</h2>')
        content.write('<div class="section inline">')
        for model_name in self.models.values():
            content.write(self.get_graph('bars/{}/Feature areas/{}/Macroareas.svg'.format(model_name, feature_area_name), model_name))
        content.write('</div>')

        content.write('<h2>Feature list</h2>')
        content.write('<div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Model</th><th>Feature</th><th>Score</th><th>Base</th><th>Score - base</th><th>Languages tested</th></tr></thead><tbody>')
        for model, model_name in self.models.items():
            for feature_name, feature in self.averages[model]['features'][feature_area_name].items():
                content.write('<tr>')
                content.write('<td>{}</td>'.format(model_name))
                content.write('<td>{}</td>'.format(feature_name))
                content.write('<td class="right">{:1.4f}</td>'.format(feature['score']))
                content.write('<td class="right">{:1.4f}</td>'.format(feature['base']))
                content.write('<td class="right">{:1.4f}</td>'.format(feature['score']-feature['base']))
                content.write('<td class="right">{}</td>'.format(feature['count']))
                content.write('</tr>')
        content.write('</tbody></table>')
        content.write('</div>')

        content.write('<h2>Feature graphs</h2>')
        content.write('<div class="section inline">')

        for model_name in self.models.values():
            content.write(self.get_graph('bars/{}/Feature areas/{}/All features.svg'.format(model_name, feature_area_name), model_name))

        for model_name in self.models.values():
            content.write(self.get_graph('count/{}/{}.svg'.format(model_name, feature_area_name), model_name))

        content.write('</div>')

        self.prepare_lang_data(True, True)

        content.write('<h2>Language list</h2>')
        content.write('<div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Language ID</th><th>Language name</th><th>Macroarea</th><th>Score</th><th>Base</th><th>Score - base</th><th>Number of features</th></tr></thead><tbody>')
        if 'languages' in self.averages['individual_languages']['feature_area'][feature_area_name]:
            for lang_id, language in self.averages['individual_languages']['feature_area'][feature_area_name]['languages'].items():
                langname, macroarea = self.langnames[lang_id]
                content.write('<tr>')
                content.write('<td>{}</td>'.format(lang_id))
                content.write('<td>{}</td>'.format(langname))
                content.write('<td>{}</td>'.format(macroarea))
                content.write('<td class="right">{:1.4f}</td>'.format(language['score']))
                content.write('<td class="right">{:1.4f}</td>'.format(language['base']))
                content.write('<td class="right">{:1.4f}</td>'.format(language['score']-language['base']))
                content.write('<td class="right">{}</td>'.format(language['count']))
                content.write('</tr>')
        content.write('</tbody></table>')
        content.write('</div>')

        content.write('<h2>Language map</h2>')
        content.write('<div class="section">')
        content.write(self.get_graph('maps/{} in all languages.svg'.format(feature_area_name)))
        content.write('</div>')

        content.write(self.javascript_tooltips(True, True))

        return content.getvalue()

    def all_features_page(self):
        content = StringIO()
        content.write('<h1>All features</h1>')
        content.write('<h2>Feature list</h2>')
        content.write('<div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Model</th><th>Feauture area</th><th>Feature</th><th>Score</th><th>Base</th><th>Score - base</th><th>Languages tested</th></tr></thead><tbody>')
        for model, model_name in self.models.items():
            for feature_area_name, feature_area in self.averages[model]['features'].items():
                for feature_name, feature in feature_area.items():
                    feature_id = feature_name.split()[0]
                    # So features can be sorted correctly
                    feature_id_num = int(''.join([char for char in feature_id if char.isdigit()]))
                    content.write('<tr>')
                    content.write('<td>{}</td>'.format(model_name))
                    content.write('<td>{}</td>'.format(feature_area_name))
                    content.write('<td data-order="{}"><a href="{}">{}</a></td>'.format(feature_id_num, feature_id, feature_name))
                    content.write('<td class="right">{:1.4f}</td>'.format(feature['score']))
                    content.write('<td class="right">{:1.4f}</td>'.format(feature['base']))
                    content.write('<td class="right">{:1.4f}</td>'.format(feature['score']-feature['base']))
                    content.write('<td class="right">{}</td>'.format(feature['count']))
                    content.write('</tr>')
        content.write('</table></div>')

        return content.getvalue()

    def individual_feature_page(self):
        feature_area_name, feature_name = self.get_feature_info_from_id(self.path[1], self.results)
        if not feature_area_name:
            return ''

        content = StringIO()
        content.write('<h1>{}</h1>'.format(feature_name))
        content.write('<h2>Summary</h2><div class="section columns">')
        for model, model_name in self.models.items():
            feature = self.averages[model]['features'][feature_area_name][feature_name]
            content.write('<table class="summary"><tr><th colspan="2">{}</th></tr>'.format(model_name))
            content.write('<tr><td>Score</td><td>{:1.4f}</td></tr>'.format(feature['score']))
            content.write('<tr><td>Base</td><td>{:1.4f}</td></tr>'.format(feature['base']))
            content.write('<tr><td>Score - base</td><td>{:1.4f}</td></tr>'.format(feature['score']-feature['base']))
            content.write('<tr><td>Languages tested</td><td>{}</td></tr></table>'.format(feature['count']))
        content.write('</div>')

        content.write('<h2>Macroarea list</h2><div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Model</th><th>Macroarea</th><th>Score</th><th>Base</th><th>Score - base</th><th>Languages tested</th></tr></thead><tbody>')
        for model, model_name in self.models.items():
            for area_name, area in self.results[model][feature_area_name][feature_name].items():
                content.write('<tr>')
                content.write('<td>{}</td>'.format(model_name))
                content.write('<td>{}</td>'.format(area_name))
                content.write('<td class="right">{:1.4f}</td>'.format(area['score']))
                content.write('<td class="right">{:1.4f}</td>'.format(area['base']))
                content.write('<td class="right">{:1.4f}</td>'.format(area['score']-area['base']))
                content.write('<td class="right">{}</td>'.format(area['amount_in_test']))
                content.write('</tr>')
        content.write('</tbody></table>')
        content.write('</div>')

        content.write('<h2>Macroarea graphs</h2><div class="section inline">')
        for model, model_name in self.models.items():
            content.write(self.get_graph('bars/{}/Feature areas/{}/features/{}.svg'.format(model_name, feature_area_name, self.path[1]), model_name))
        content.write('</div>')

        return content.getvalue()

    def macroarea_average_page(self):
        content = StringIO()

        content.write('<h1>Macroareas</h1>')
        content.write('<h2>Macroarea list</h2>')
        content.write('<div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Model</th><th>Macroarea</th><th>Score</th><th>Base</th><th>Score - base</th><th>Languages tested</th></tr></thead><tbody>')
        for model, model_name in self.models.items():
            for area_name, area in self.averages[model]['area'].items():
                content.write('<tr>')
                content.write('<td>{}</td>'.format(model_name))
                content.write('<td>{}</td>'.format(area_name))
                content.write('<td>{:1.4f}</td>'.format(area['score']))
                content.write('<td>{:1.4f}</td>'.format(area['score']))
                content.write('<td>{:1.4f}</td>'.format(area['score']-area['base']))
                content.write('<td>{}</td>'.format(area['count']))
                content.write('</tr>')
        content.write('</tbody></table></div>')

        content.write('<h2>Macroarea graphs</h2><div class="section inline">')
        for model, model_name in self.models.items():
            content.write(self.get_graph('bars/{}/Macroareas.svg'.format(model_name), model_name))
        content.write('</div>')

        return content.getvalue()

    def individual_macroarea_page(self):
        area_name = self.capitalize_first_letters(self.path[1])
        if not area_name in self.averages['individual_languages']['area']:
            return ''

        self.prepare_lang_data(True, False)

        content = StringIO()

        content.write('<h1>{}</h1>'.format(area_name))

        content.write('<h2>Summary</h2><div class="section columns">')
        for model, model_name in self.models.items():
            if area_name in self.averages[model]['area']:
                area = self.averages[model]['area'][area_name]
                content.write('<table class="summary"><tr><th colspan="2">{}</th></tr>'.format(model_name))
                content.write('<tr><td>Score</td><td>{:1.4f}</td></tr>'.format(area['score']))
                content.write('<tr><td>Base</td><td>{:1.4f}</td></tr>'.format(area['base']))
                content.write('<tr><td>Score - base</td><td>{:1.4f}</td></tr>'.format(area['score']-area['base']))
                content.write('<tr><td>Languages tested</td><td>{}</td></tr></table>'.format(area['count']))
        content.write('</div>')

        content.write('<h2>Feature area list</h2><div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Model</th><th>Feature area</th><th>Score</th><th>Base</th><th>Score - base</th><th>Langauges tested</th></tr></thead><tbody>')
        for model, model_name in self.models.items():
            for feature_area_name, feature_area in self.averages[model]['feature_area'].items():
                if 'areas' in feature_area and area_name in feature_area['areas']:
                    area = feature_area['areas'][area_name]
                    content.write('<tr>')
                    content.write('<td>{}</td>'.format(model_name))
                    content.write('<td>{}</td>'.format(feature_area_name))
                    content.write('<td class="right">{:1.4f}</td>'.format(area['score']))
                    content.write('<td class="right">{:1.4f}</td>'.format(area['base']))
                    content.write('<td class="right">{:1.4f}</td>'.format(area['score']-area['base']))
                    content.write('<td class="right">{}</td>'.format(area['count']))
                    content.write('</tr>')
        content.write('</tbody></table></div>')

        content.write('<h2>Feature area graphs</h2><div class="section inline">')
        for model_name in self.models.values():
            content.write(self.get_graph('bars/{}/Macroareas/{}.svg'.format(model_name, area_name), model_name))
        content.write('</div>')

        content.write('<h2>Language list</h2><div class="section">')
        content.write('<table class="datatable"><thead><tr><th>Language ID</th><th>Language name</th><th>Score</th><th>Base</th><th>Score - base</th><th>Languages tested</th></tr></thead><tbody>')
        for lang_id, language in self.averages['individual_languages']['languages'].items():
            langname, macroarea = self.langnames[lang_id]
            if macroarea == area_name:
                content.write('<tr>')
                content.write('<td>{}</td>'.format(lang_id))
                content.write('<td>{}</td>'.format(langname))
                content.write('<td class="right">{:1.4f}</td>'.format(language['score']))
                content.write('<td class="right">{:1.4f}</td>'.format(language['base']))
                content.write('<td class="right">{:1.4f}</td>'.format(language['score'] - language['base']))
                content.write('<td class="right">{}</td>'.format(language['count']))
                content.write('</tr>')
        content.write('</tbody></table></div>')

        content.write('<h2>Language map</h2>')
        content.write('<div class="section">')
        content.write(self.get_graph('maps/{}/Average.svg'.format(area_name)))
        content.write('</div>')

        content.write(self.javascript_tooltips(True, False))

        return content.getvalue()

    def javascript_tooltips(self, langs = True, features = True):
        code = StringIO()
        code.write('<script>')
        code.write('$(document).ready(function() {')
        if langs:
            for lang_id in self.averages['individual_languages']['languages'].keys():
                lang_name, _ = self.langnames[lang_id]
                code.write('$("#{}").tooltip({{content: "{}", items: "#{}", show: "slideDown"}});'.format(lang_id, lang_name, lang_id))
        if features:
            for feature_id, feature_name in self.features.items():
                code.write('$("[id=\'{}\']").tooltip({{content: "{}", items: "#{}", show: "slideDown"}});'.format(feature_id, feature_name, feature_id))
                code.write('$("[id=\'{}\']").click(function() {{window.location.href = "/"+session+"/feature/{}";}}).addClass("pointer");'.format(feature_id, feature_id))
        code.write('});')
        code.write('</script>')
        return code.getvalue()


    def unknown_page(self):
        return "Could not find the selected page"

    def capitalize_first_letters(self, string):
        '''
        Capitalizes the first letter of every word and replaces '-' with spaces.
        For example 'nominal-categories' turns into 'Nominal Categories'
        '''
        return ' '.join(map(lambda x: x.capitalize(), string.split('-')))

    def urlify_string(self, string):
        '''
        Does the opposite of the capitalize_first_letters() function
        '''
        return string.replace(' ', '-').lower()


    def get_graph(self, file, title = '', text = ''):
        file = 'temp/{}/graphs/{}'.format(self.session, file)
        if not os.path.isfile(file):
            return ''
        content = StringIO()
        content.write('<div class="graph">')
        if title:
            content.write('<h3>{}</h3>'.format(title))
        if text:
            content.write('<p>{}</p>'.format(text))
        with open(file, encoding='utf-8') as f:
            content.write(f.read())
        content.write('</div>')
        return content.getvalue()

    def get_feature_info_from_id(self, feature_ID, results):
        feature_ID = feature_ID.upper()
        for feature_area_name, feature_area in self.results['individual_languages'].items():
            for feature_name in feature_area.keys():
                fid = feature_name.split()[0].upper()
                if feature_ID == fid:
                    return (feature_area_name, feature_name)
        return ('', '')

    def create_menu(self):
        content = StringIO()
        content.write('<div id="menu"><ul>')
        selected = ' class="selected"' if not self.path else ''
        content.write('<li><a href="/{}/"{}>Summary</a></li>'.format(self.session, selected))
        selected = ' class="selected"' if self.path and 'feature-area' == self.path[0] else ''
        content.write('<li><a href="/{}/feature-area/"{}>Feature areas</a>'.format(self.session, selected))
        content.write('<ul>')
        for feature_area_name in self.results['individual_languages'].keys():
            content.write('<li><a href="/{}/feature-area/{}/">{}</a></li>'.format(self.session, self.urlify_string(feature_area_name), feature_area_name))
        content.write('</ul></li>')
        selected = ' class="selected"' if self.path and 'feature' == self.path[0] else ''
        content.write('<li><a href="/{}/feature/"{}>Features</a></li>'.format(self.session, selected))
        selected = ' class="selected"' if self.path and 'languages' == self.path[0] else ''
        content.write('<li><a href="/{}/languages/"{}>Languages</a></li>'.format(self.session, selected))
        selected = ' class="selected"' if self.path and 'macroarea' == self.path[0] else ''
        content.write('<li><a href="/{}/macroarea/"{}>Maroareas</a>'.format(self.session, selected))
        content.write('<ul>')
        for area_name in self.averages['individual_languages']['area'].keys():
            content.write('<li><a href="/{}/macroarea/{}/">{}</a></li>'.format(self.session, self.urlify_string(area_name), area_name))
        content.write('</ul></li>')

        content.write('</ul></div>')

        return content.getvalue()

    def prepare_lang_data(self, langnames = True, features = True):
        '''
        Prepares data from the language.csv file.
        '''
        with open('language.csv', 'rt', encoding='utf8') as file:
            reader = csv.reader(file)
            languages = list(reader)
        # headers
        headers = languages.pop(0)
        if langnames:
            self.langnames = {lang[1]: (lang[3], lang[8]) for lang in languages}
        if features:
            self.features = {feature.split()[0]: feature for feature in headers[10:]}
