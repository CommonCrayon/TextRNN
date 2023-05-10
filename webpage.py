# Importing modules
from flask import Flask, request
import tensorflow as tf
import numpy as np

# Load the SavedModel
model = tf.saved_model.load('./my_model')

# Define the CSS styles
styles = '''
    body {
        background-color: #f5f5f5;
        font-family: 'Helvetica Neue', sans-serif;
    }

    h1 {
        font-size: 3em;
        color: #333;
        text-align: center;
        margin-top: 2em;
    }

    form {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-top: 3em;
    }

    input[type="text"] {
        font-size: 1em;
        padding: 0.5em;
        border: 2px solid #ccc;
        border-radius: 4px;
        margin-bottom: 1em;
        width: 100%;
        max-width: 600px;
        height: 10em;
        overflow: auto;
    }

    input[type="submit"] {
        font-size: 1em;
        padding: 0.5em;
        border: none;
        border-radius: 4px;
        background-color: #333;
        color: #fff;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        width: 100%;
        max-width: 600px;
    }

    input[type="submit"]:hover {
        background-color: #555;
    }

    p {
        font-size: 1.2em;
        font-weight: bold;
        margin-top: 2em;
        text-align: center;
        color: #333;
    }

    footer {
        background-color: #333;
        color: white;
        padding: 1em;
        text-align: center;
        position: fixed;
        bottom: 0;
        width: 100%;
    }
'''


# Create a Flask app
app = Flask(__name__)

# Define a Flask route for the prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    # When the Predict button has been submitted
    if request.method == 'POST':
        # Get the text data from the POST request
        data = request.form['data']

        # Use the loaded model to make predictions on the input data
        prediction = model(tf.constant([data]))[0]

        # Convert the prediction to a label
        label = np.argmax(prediction)

        if (label == 1):
            response = "True"
        else:
            response = "False"

        return f'''
            <html>
                <head>
                    <title>Fake vs Real News Headline Predictor</title>
                    <style>{styles}</style>
                </head>
                <body>
                    <div>
                        <h1>Fake vs Real News Headline Predictor</h1>
                        <hr>
                        <form method="post">
                            <input type="text" name="data" placeholder="Enter a news headline...">
                            <input type="submit" value="Predict">
                        </form>
                        <p>Input: {data}</p>
                        <p>Prediction: {response}</p>
                    </div>
                    <footer>
                        <p style="color:white">&copy; Made by Group 12 | NLP UTS 2023</p>
                    </footer>
                </body>
            </html>
        '''
    

    # Return the webpage when a form request hasn't been sent
    return f'''
        <html>
            <head>
                <title>Fake vs Real News Headline Predictor</title>
                <style>{styles}</style>
            </head>
            <body>
                <div>
                    <h1>Fake vs Real News Headline Predictor</h1>
                    <hr>
                    <form method="post">
                        <input type="text" name="data" placeholder="Enter a news headline...">
                        <input type="submit" value="Predict">
                    </form>
                </div>
                <footer>
                        <p style="color:white">&copy; Made by Group 12 | NLP UTS 2023</p>
                </footer>
            </body>
        </html>
    '''


# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
