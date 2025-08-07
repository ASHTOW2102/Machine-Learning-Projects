# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Sample Data
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# X = df[['Hours_Studied']]
# y = df['Scores']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted = None
    if request.method == "POST":
        year_of_exp = float(request.form.get("YearsExperience"))
        predicted = model.predict([[year_of_exp]])[0]
    return render_template("index.html", predicted=predicted)

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=5000, debug=True)
