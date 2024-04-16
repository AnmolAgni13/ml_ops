from flask import Flask, request,render_template
import joblib

#Load the pre-trained model
model=joblib.load('C:\\Users\\SERVER\\Desktop\\MLOPS\\IRIS-model.pkl')

app=Flask(__name__, template_folder='index.html')
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    feature1=float(request.form['feature1'])
    feature2=float(request.form['feature2'])
    feature3=float(request.form['feature3'])
    feature4=float(request.form['feature4'])
    prediction = model.predict([[feature1,feature2,feature3,feature4]])

    return render_template("index.html",prediction=prediction[0])

if __name__=='__main__':
    app.run(debug=True)