"""
This code shows a async way in Flask of submiting POST request 
via a form element on frontend(client) that dynamically renders new content 
on the same page index.html
"""
from flask import Flask, jsonify, redirect, render_template, request, url_for, redirect
import pandas as pd
import numpy as np
from linear_regression import LinearRegression

app = Flask(__name__)

df = pd.read_csv('real_estate.csv', names=['price', 'rl_size', 'year'], skiprows=1)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route('/output', methods=['GET', 'POST'])
def output():
    data = float(request.json['user_input'])
    print('Data: ', data)
    model = LinearRegression()
    model.fit(df.rl_size, df.price)
    predictions = model.predict(np.array(data))
    print(predictions)
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)