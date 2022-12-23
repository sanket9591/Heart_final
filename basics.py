from flask import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True,host='127.0.0.2',port=3031)