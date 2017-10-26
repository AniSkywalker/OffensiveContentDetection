from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
import os, sys, inspect

sys.path.append('../../')

from OffensiveContentDetection.src.twitter_module import twitter_actions
interaction = twitter_actions.Interaction()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_data', methods=['POST'])
def handle_data():
    userHandle = request.form['userTwitterHandle']
    if len(userHandle) > 1 and userHandle[0] == '@':
        return redirect('/user/'+userHandle)

@app.route('/user/<user_name>')
def page(user_name):
    interaction.get_recent_tweets(user_name)
    return render_template('index.html', username=user_name)
