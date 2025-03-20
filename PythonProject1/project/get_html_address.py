from flask import Flask
from flask import render_template


app = Flask(__name__) # __name__ 主线程名称
@app.route("/hello")
def hello():
    return render_template("index.html")




@app.route("/hi")
def hi():
    return"<h1 style='color:red;'>Hi</h1>"

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=9000,debug=True)
