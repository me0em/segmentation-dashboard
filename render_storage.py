from json2html import *
import tempfile
import webbrowser
import os
import json

def view(report):
    html = json2html.convert(json=report, table_attributes='border="1px"; border-color: rgba(0,0,255,0.25)')

    html = f'<link rel="stylesheet" href="{os.getcwd()}/table.css">' + \
     '<h1><center>Dashboard 📊</center></h1>' + \
     '<div class="table-wrapper"><table class="fl-table">' + \
     html + \
     '</table></div>'

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tf:
        tf.write(html)
        webbrowser.open('file://' + os.path.realpath(tf.name))
        
if __name__ == "__main__":
    with open("storage.json", "r") as json_file:
        json_decoded = json.load(json_file)
        
    view(json_decoded)