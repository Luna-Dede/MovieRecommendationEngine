from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():

    movie_sample = pd.read_csv("data/sample.csv")
    sub = movie_sample[['title', 'poster_path']]

    #convert to json
    movies_json = sub.to_json(orient="records")


    return render_template('index.html', movies=movies_json)

if __name__ == '__main__':
    app.run(debug=True)
