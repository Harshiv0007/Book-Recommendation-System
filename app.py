from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

data = pickle.load(open('topfifty.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html',
                           title=list(data['title'].values),
                           author=list(data['author'].values),
                           coverImg=list(data['coverImg'].values),
                           description=list(data['description'].values),
                           likedPercent=list(data['likedPercent'].values)
                           )


if __name__ == '__main__':
    app.run(debug=True)
