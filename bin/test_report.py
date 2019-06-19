#!/usr/bin/env python

"""
This script produces an HTML report from a JSON test result file.
Use "python3 test_report.py -h" for help
"""

import json
import argparse

from os import path
from http.server import BaseHTTPRequestHandler, HTTPServer


def create_session_handler():
    class SessionHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path == "favico.ico":
                    return
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(data_doc.encode())
                    return
                if self.path.startswith('/wav/'):
                    wav_path = data[int(self.path[5:])]['wav_filename']
                    self.send_response(200)
                    self.send_header('Content-type', 'audio/wav')
                    self.send_header('Content-length', path.getsize(wav_path))
                    self.end_headers()
                    with open(wav_path, 'rb') as wav_file:
                        self.wfile.write(wav_file.read())
                    return
            except IOError:
                self.send_error(404)
    return SessionHandler

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Serve or write a test report page')
    PARSER.add_argument('test_output_file', help='Test output file (--test_output_file parameter of evaluate.py)')
    PARSER.add_argument('--port', help='Port for serving report page on localhost.')
    PARSER.add_argument('--save_to', help='Just saves HTML report to provided file path for offline use. No serving.')
    PARSER.add_argument('--compare_with', help='Generate comparison report using provided second test output file')
    PARSER.add_argument('--exclude_wer_0', action='store_true', help='Excludes samples with WER == 0 (in comparison reports both WERs have to be 0)')
    PARSER.add_argument('--open_page', action='store_true', help='Automatically opens page in default browser')
    PARAMS = PARSER.parse_args()

    data_doc = """
    <html><head>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.19/css/jquery.dataTables.css">
    <style>
        span.file { color: darkgray; font-size: xx-small; font-family: monospace; }
        span.src { color: gray; }
        span.left { color: blue; }
        span.right { color: green; }
        span.number { font-family: monospace; }
    </style>
    <script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.10.19/js/jquery.dataTables.js"></script>
    <script type="text/javascript">
        var data = window.data = REPLACE_WITH_DATA;

        function renderNumber (value, type, row, meta) {
            return '<span class="number">' + value.toFixed(2) + '</span>';
        }
        
        function renderPercent (value, type, row, meta) {
            return '<span class="number">' + (value * 100).toFixed(2) + '</span>';
        }
    
        $(document).ready(function() {
            $('#samples').DataTable({
                paging: true,
                data: data,
                columns: [
                    {
                        title: "Sample",
                        data: "REPLACE_WITH_SRC",
                        render: function (value, type, row, meta) {
                            return '<audio controls preload="none" src="' + value + '" />';
                        },
                        width: "0"
                    },
                    REPLACE_WITH_COLUMNS
                ]
            });
        });
    </script>
    </head>
    <body>
    <table id="samples" class="compact display" style="width:100%"></table>
    </body>
    </html>
    """

    if PARAMS.compare_with:
        data_doc = data_doc.replace('REPLACE_WITH_COLUMNS', """
            {
                title: "Transcript",
                data: "src",
                render: function (value, type, row, meta) {
                    return '<span class="file">' + row.wav_filename + '</span><br/>' + 
                           '<span class="src">' + row.src + '</span><br/>' + 
                           '<span class="left">' + row.res + '</span><br/>' + 
                           '<span class="right">' + row.res_other + '</span>';
                }
            },
            {
                title: '<span class="left">Loss</span>',
                data: "loss",
                className: 'dt-body-right',
                render: renderNumber
            },
            {
                title: '&Delta;',
                data: "loss_delta",
                className: 'dt-body-right',
                render: renderNumber
            },
            {
                title: '<span class="right">Loss</span>',
                data: "loss_other",
                className: 'dt-body-right',
                render: renderNumber
            },
            {
                title: '<span class="left">CER</span>',
                data: "cer",
                className: 'dt-body-right',
                render: renderPercent
            },
            {
                title: '&Delta;',
                data: "cer_delta",
                className: 'dt-body-right',
                render: renderPercent
            },
            {
                title: '<span class="right">CER</span>',
                data: "cer_other",
                className: 'dt-body-right',
                render: renderPercent
            },
            {
                title: '<span class="left">WER</span>',
                data: "wer",
                className: 'dt-body-right',
                render: renderPercent
            },
            {
                title: '&Delta;',
                data: "wer_delta",
                className: 'dt-body-right',
                render: renderPercent
            },
            {
                title: '<span class="right">WER</span>',
                data: "wer_other",
                className: 'dt-body-right',
                render: renderPercent
            }
        """)
    else:
        data_doc = data_doc.replace('REPLACE_WITH_COLUMNS', """
            {
                title: "Transcript",
                data: "src",
                render: function (value, type, row, meta) {
                    return '<span class="file">' + row.wav_filename + '</span><br/>' + 
                           '<span class="src">' + row.src + '</span><br/>' + 
                           '<span class="left">' + row.res + '</span>';
                }
            },
            {
                title: 'Loss',
                data: "loss",
                className: 'dt-body-right',
                render: renderNumber
            },
            {
                title: "CER",
                data: "cer",
                className: 'dt-body-right',
                render: renderPercent
            },
            {
                title: "WER",
                data: "wer",
                className: 'dt-body-right',
                render: renderPercent
            }
        """)

    data = None
    with open(PARAMS.test_output_file, 'r') as data_file:
        data = json.loads(data_file.read())

    if PARAMS.compare_with:
        lookup = {}
        for entry in data:
            lookup[entry['wav_filename']] = entry
        data = []
        with open(PARAMS.compare_with, 'r') as other_file:
            other = json.loads(other_file.read())
            for other_entry in other:
                if other_entry['wav_filename'] in lookup:
                    entry = lookup[other_entry['wav_filename']]
                    entry['res_other'] = other_entry['res']
                    entry['loss_other'] = other_entry['loss']
                    entry['cer_other'] = other_entry['cer']
                    entry['wer_other'] = other_entry['wer']
                    entry['loss_delta'] = other_entry['loss'] - entry['loss']
                    entry['cer_delta'] = other_entry['cer'] - entry['cer']
                    entry['wer_delta'] = other_entry['wer'] - entry['wer']
                    del lookup[other_entry['wav_filename']]
                    if not PARAMS.exclude_wer_0 or entry['wer'] > 0 or other_entry['wer'] > 0:
                        data.append(entry)
    else:
        if PARAMS.exclude_wer_0:
            data = [sample for sample in data if sample['wer'] > 0]

    if not PARAMS.save_to:
        for i, entry in enumerate(data):
            entry['wav_href'] = '/wav/' + str(i)

    data_doc = data_doc.replace('REPLACE_WITH_SRC', 'wav_filename' if PARAMS.save_to else 'wav_href')
    data_doc = data_doc.replace('REPLACE_WITH_DATA', json.dumps(data))

    if PARAMS.save_to:
        with open(PARAMS.save_to, 'w') as report:
            report.write(data_doc)
    else:
        try:
            port = int(PARAMS.port) if PARAMS.port else 8080
            server = HTTPServer(('localhost', port), create_session_handler())
            print('Started serving report on http://localhost:%s' % port)
            server.serve_forever()
        except KeyboardInterrupt:
            print('^C received, shutting down the web server')
        server.socket.close()

