"""
This code shows a typical way in Flask of submiting POST request 
via a form element on frontend(client) that renders a new page
in this case output.html
"""
from flask import Flask, redirect, render_template, request, url_for, redirect
import pandas as pd
import numpy as np
from linear_regression import LinearRegression

app = Flask(__name__)

df = pd.read_csv('real_estate.csv', names=['price', 'rl_size', 'year'], skiprows=1)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = float(request.form['name'])
        model = LinearRegression()
        model.fit(df.rl_size, df.price)
        predictions = model.predict(np.array(data))

        return redirect(url_for('output', predictions=predictions))

    return render_template("index.html")


@app.route('/output', methods=['GET', 'POST'])
def output():
    predictions = request.args.get('predictions')
    return render_template("output.html", predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)