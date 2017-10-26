from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
import os, sys, inspect, numpy

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
    tweet_emotions = interaction.get_recent_tweets(user_name[1:])
    direct_emotions = interaction.get_direct_tweets(user_name[1:])
    hate = numpy.round(numpy.average(direct_emotions[0]), decimals=2)
    not_hate = 1 - hate
    offensive = numpy.round(numpy.average(direct_emotions[1]), decimals=2)
    not_offensive = 1 - offensive
    both = numpy.round(numpy.average(direct_emotions[2]), decimals=2)
    not_both = 1 - both
    return render_template('index.html', username=user_name, tweet_emotions=tweet_emotions, direct_emotions=direct_emotions, hate=hate, not_hate=not_hate, offensive=offensive, not_offensive=not_offensive, both=both, not_both=not_both)
