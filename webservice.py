from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Timer
from multiprocessing import Process
from io import StringIO, BytesIO
import time
import re
import string
import random
import os
import shutil
import pickle

from evaluate import run_evaluation
from evaluation_page import EvaluationPage

HOST_NAME = 'localhost'
PORT_NUMBER = 80

class MyHandler(BaseHTTPRequestHandler):

    def do_HEAD(self, redirect = '', mime = 'text/html; charset=utf-8', status = 200):
        self.send_response(status)
        if redirect:
            self.send_header('Location',  'http://' + HOST_NAME + '/' + redirect)
        self.send_header('Content-type', mime)
        self.end_headers()

    def parse_post_data(self):
        boundary = self.headers['Content-type'].split('boundary=')[1].encode()
        content_length = int(self.headers['Content-Length'])

        line = self.rfile.readline()
        content_length -= len(line)

        if not boundary in line:
            return (False, "Can't find content");

        feature_groups = []
        features = []
        classifier = ''
        knnk = 0

        while content_length > 0:
            line = self.rfile.readline()
            content_length -= len(line)
            name = re.findall(r'name="([^"]*)"', line.decode('utf-8'))

            if not name:
                return (False, "Could not find name of POST variable")

            name = name[0].split('-')

            if name[0] == 'featuregroups':
                # It will always store 'on' in the content, so we don't need to
                # load it
                _, content_length = self.get_post_val(content_length)
                feature_groups.append(int(name[1]))

            elif name[0] == 'features':
                data, content_length = self.get_post_val(content_length)
                if data:
                    features = data.split(',')

            elif name[0] == 'classifier':
                classifier, content_length = self.get_post_val(content_length)
                if classifier not in ['knn', 'svm', 'mlp']:
                    return (False, "Unknown classifier")

            elif name[0] == 'knnk':
                data, content_length = self.get_post_val(content_length)
                knnk = int(data)
                if knnk < 0 or knnk > 50:
                    return (False, "Invalid k")

            elif name[0] == 'embeddings':
                # Check if it's a pickle file
                file_name = re.findall(r'filename="([^"]*)"', line.decode('utf-8'))[0]
                # To get the extension
                file_name = file_name.split(".")
                if len(file_name) != 2 or file_name[1].upper() != 'PKL':
                    return (False, "Invalid file. File must be a pickle")
                # Content type
                content_length -= len(self.rfile.readline())
                # Blank line
                content_length -= len(self.rfile.readline())
                # First data line
                line = self.rfile.readline()
                content_length -= len(line)

                embeddings_pickle = BytesIO()

                while not boundary in line:
                    if content_length <= 0:
                        return (False, "Unexpected end of stream")
                    embeddings_pickle.write(line)
                    line = self.rfile.readline()
                    content_length -= len(line)

                try:
                    embeddings_pickle.seek(0)
                    embeddings = pickle.load(embeddings_pickle)
                except pickle.UnpicklingError:
                    return (False, "Invalid pickle file")
            else:
                return (False, "Unexpected POST value")

        data = {'embeddings' : embeddings,
                'feature_groups': feature_groups,
                'features': features,
                'classifier': classifier,
                'knnk': knnk}

        return (True, data)

    def get_post_val(self, content_length):
        # Blank line
        content_length -= len(self.rfile.readline())
        # Data
        line = self.rfile.readline()
        content_length -= len(line)
        # The boundary
        content_length -= len(self.rfile.readline())
        # Last 2 characters are always \r\n
        data = line.decode('utf-8')[0:-2]
        return (data, content_length)

    def do_POST(self):
        session = gen_session()

        # In the unlikely event that the session is taken
        while os.path.isdir('temp/'+session):
            session = gen_session()

        success, content = self.parse_post_data()

        if success:
            os.makedirs('temp/'+session)
            process = Process(target = start_evaluation, args = (content, session))
            process.start()
            # 303 is 'see other' redirection
            self.do_HEAD(redirect = session + '/', status = 303)
            #self.wfile.write(bytes("OK. Loading", 'UTF-8'))
        else:
            page_content = StringIO()
            page_content.write(self.front_page())
            page_content.write('<p class="error-message">' + content + '</p>')
            self.page(page_content.getvalue())


    def do_GET(self):
        if self.path == '/':
            self.page(self.front_page())
            return

        # Just to be compliant with the URL standards, in case parameters are
        # passed with the URL
        url = self.path.split('?')

        # If requestion a file
        if '.' in url[0]:
            self.send_file(url[0])
            return

        path = url[0].split('/')
        # Remove empty elements from list
        path = list(filter(None, path))

        session = path[0]
        session_dir = 'temp/' + session + '/'
        if os.path.isdir(session_dir):
            is_ready = os.path.isfile(session_dir + 'data.pkl')
            # If sending ajax call to check if done
            if len(path) == 2 and path[1] == 'check':
                self.do_HEAD()
                if is_ready:
                    self.wfile.write(bytes("ok", 'UTF-8'))
                else:
                    self.wfile.write(bytes("no", 'UTF-8'))
            # Else if normal page call
            elif is_ready and len(path) == 1:
                self.do_HEAD(redirect = '{}/summary'.format(session), status = 303)
            elif is_ready:
                eval_page = EvaluationPage(path)
                eval_page.create_page()
                self.page(eval_page.content.getvalue(), eval_page.status)
            else:
                self.loading_page()
        else:
            self.page('<p>Could not find the session. Perhaps the session has expired?</p><p><a href="/">Go back</a></p>', 404)

    def send_file(self, path):
        mime = ''

        # Removing the first / and fixing spaces
        file = path[1:].replace('%20', ' ')

        if file.endswith('.js'):
            mime = 'application/javascript'
        elif file.endswith('.css'):
            mime = 'text/css'
        elif file.endswith('.png'):
            mime = 'image/png'
        elif file.endswith('.ico'):
            mime = 'image/x-icon'

        if mime and os.path.isfile(file):
            self.do_HEAD(mime = mime)
            with open(file, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.page("Couldn't find the requested file", 404)

    def front_page(self):
        content = StringIO()
        feature_groups = ['Phonology', 'Morphology',
                             'Nominal Categories', 'Nominal Syntax',
                             'Verbal Categories', 'Word Order',
                             'Simple Clauses', 'Complex Sentences',
                             'Lexicon', 'Sign Languages', 'Other', 'Word Order']
        content.write('<div id="mainform">')
        content.write('<h1>Evaluation of language embeddings</h1>')
        content.write('''<p>This tool allows you to upload and evaluate language
                         representations. The embeddings must be a Python pickle,
                         where the keys are language IDs (ISO 639-3) and the values
                         are language embeddings, which must constist of floating
                         lists or numpy arrays. When submiting the embeddings,
                         the evaluation takes a couple of minutes before it is ready.
                         Once the results are ready, you will be redirected to the
                         results page. You will be given a session ID in the URL,
                         so you can bookmark the page and come back later. A session is valid for
                         a week, after which the embeddings must be resubmitted if you
                         wish to view the results again. You can read more about this
                         project on its Github page at
                         <a href="https://github.com/peterspliid/Language-evaluation"
                         target="_blank">https://github.com/peterspliid/Language-evaluation</a>.</p>''')
        content.write('<form method="post" enctype="multipart/form-data" target="_blank">')
        content.write('<table></tr><th>Feature areas</th><td>')
        for i, feature_group in enumerate(feature_groups):
            content.write('<p><input type="checkbox" name="featuregroups-'+str(i+1)+'" id="fg'+str(i+1)+'" checked />')
            content.write('<label for="fg'+str(i+1)+'">'+feature_group+"</label></p>")
        content.write('''
            </td></tr>
            <tr>
                <th>
                    <label for="features">Features<p>Comma seperated IDs from WALS</p></label>
                </th>
                <td>
                    <input type="text" name="features" id="features" />
                </td>
            </tr>
            <tr>
                <th>
                    <label for="embeddings">Language embeddings</label>
                </th>
                <td>
                    <input type="file" name="embeddings" id="embeddings" accept=".pkl" />
                </td>
            </tr>
            <tr>
                <th>
                    <label for="classifier">Classifier</label>
                </th>
                <td>
                    <select name="classifier" id="classifier">
                        <option value="knn">k-nearest neighbors</option>
                        <option value="svm">Support vector machine</option>
                        <option value="mlp">Multilayer perceptron</option>
                    </select>
                </td>
            </tr>
            <tr>
                <th>
                    <label for="knnk">k for k-nearest neighbors</label>
                </th>
                <td>
                    <input type="number" name="knnk" id="knnk" value="17" accept=".pkl" min="1" max="50" />
                </td>
            </tr>
            </table>
            <input type="submit" value="Submit" />
            </form></div>
        ''')
        return content.getvalue()

    def loading_page(self):
        content = StringIO()
        content.write('<script src="/js/loading.js?ver=1.0"></script>')
        content.write('<div id="loading-page">')
        content.write('<!-- https://loading.io/css/ -->')
        content.write('<div class="lds-facebook"><div></div><div></div><div></div></div>')
        content.write('<h1>Please wait</h1>')
        content.write('<h2>Your embeddings are being evaluated</h2>')
        content.write('<p>This will take a couple of minutes. You will automatically be redirected when the results are ready.</p>')
        content.write('</div>')
        self.page(content.getvalue())

    def page(self, content, status = 200):
        page = StringIO()
        self.do_HEAD(status = status)
        page.write(self.header())
        page.write(content)
        page.write(self.footer())
        self.wfile.write(bytes(page.getvalue(), 'utf-8'))

    def header(self):
        return '''
            <html>
            <head>
            <title>Evaluation of language representations</title>
            <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
            <script type="text/javascript" src="/js/jquery-ui.min.js"></script>
            <script type="text/javascript" src="/js/script.js"></script>
            <script type="text/javascript" src="https://cdn.datatables.net/v/ju/dt-1.10.18/datatables.min.js"></script>
            <link rel="stylesheet" type="text/css" href="/css/jquery-ui.min.css">
            <link rel="stylesheet" type="text/css" href="/css/style.css?ver=1.0">
            <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/v/ju/dt-1.10.18/datatables.min.css"/>
            </head>
            <body>
        '''
    def footer(self):
        return '</body></html>'

def start_evaluation(args, session):
    data = run_evaluation(args['embeddings'], False, True, args['knnk'],
        args['classifier'], args['feature_groups'], args['features'],
        'temp/'+session+'/')

    with open('temp/'+session+'/data.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def gen_session():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))

def clean_folder(folder, del_self = False):
    if del_self:
        try:
            shutil.rmtree(folder)
        except Exception as e:
            print(e)
    else:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

def clean_temp():
    '''
    Deletes anything in the temp folder, that is older than 1 week. Runs once
    every hour
    '''
    Timer(3600, clean_temp).start()
    for the_file in os.listdir('temp/'):
        path = os.path.join('temp', the_file)
        folder_time_accessed = os.stat(path).st_atime
        folder_age = time.time() - folder_time_accessed
        days_old = folder_age / 60 / 60 / 24
        # If older than a week
        if days_old > 7:
            clean_folder(path, True)


if __name__ == '__main__':
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    clean_temp()

    if not os.path.isdir('temp'):
        os.makedirs('temp')
    #else:
    #    clean_folder('temp/')

    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
    clean_folder('temp/')