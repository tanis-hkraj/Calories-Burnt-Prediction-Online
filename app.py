from flask import Flask,render_template, request
import pickle
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.form.get('age')=="1":
        gender=1
    else:
        gender=0
    X_List=[gender,request.form.get('age'),request.form.get('height'),request.form.get('weight'),request.form.get('duration'),request.form.get('heart'),request.form.get('body')]
    X_df=pd.DataFrame([X_List])
    X_df.columns=['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
    output=model.predict(X_df)[0]
    return render_template("index.html",prediction_text="Congratulations!!! You have burnt {:.3f} Calories today.".format(output))
    
       
if __name__=="__main__":
    app.run(debug=True)  
    


