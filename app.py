from flask import Flask, render_template, request
from datetime import datetime
import re

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template( "index.html")

@app.route("/", methods=['POST'])
def predict():
    imagefile = request.files['imagefiles']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    return render_template( "index.html")

if __name__ == '__main__':
    app.run(debug=True)