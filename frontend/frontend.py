from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_data', methods=['POST'])
def handle_data():
    userHandle = request.form['userTwitterHandle']
    return redirect('/user/'+userHandle)

@app.route('/user/<user_name>')
def page(user_name):
    return render_template('index.html', username = user_name)
