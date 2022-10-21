"""This is the entire Flask app hosted in Cloud Run"""
import logging
import flask

# Initialise flask application and logging
app = flask.Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['POST', 'GET'])
def compile_pdf():
   return "lol"
