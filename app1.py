import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('slmmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods =['POST'])
def predict():
     
    int_features = float(request.form['experience'])
    
    features = np.reshape(int_features, (1, -1))
    prediction = model.predict(features)
    
    output = prediction
    
    return render_template('index.html',prediction_text="For this experience you will get {} this of much salary".format(output))

if __name__ == "__main__":
    
    app.run(debug=True)
    
        