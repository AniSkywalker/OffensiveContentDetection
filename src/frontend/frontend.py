from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, jsonify
import os, sys, inspect, numpy

sys.path.append('../../')

from src.twitter_module import twitter_actions

interaction = twitter_actions.Interaction()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/handle_data', methods=['POST'])
def handle_data():
    userHandle = request.form['userTwitterHandle']
    if len(userHandle) > 1 and userHandle[0] == '@':
        return redirect('/user/' + userHandle)


@app.route('/user/<user_name>')
def predict_hate_emotion(user_name):
    tweet_emotions = interaction.get_recent_tweets(user_name[1:])
    print(tweet_emotions)
    direct_emotions = interaction.get_direct_tweets(user_name[1:])
    hate = numpy.round(numpy.average(direct_emotions[0]), decimals=2)

    not_hate = 1 - hate
    offensive = numpy.round(numpy.average(direct_emotions[1]), decimals=2)
    not_offensive = 1 - offensive
    both = numpy.round(numpy.average(direct_emotions[2]), decimals=2)
    not_both = 1 - both
    return render_template('index.html', username=user_name, tweet_emotions=tweet_emotions,
                           direct_emotions=direct_emotions, hate=hate, not_hate=not_hate, offensive=offensive,
                           not_offensive=not_offensive, both=both, not_both=not_both)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['input_tweet']
    score = interaction.get_sarcasm_score(text)
    print(score)
    return render_template('index.html',input_text=text, sarcasm_score=score)


@app.route('/generation/')
def generation(text):
    g_text = text + 'I will update'
    return render_template('index.html', generated_text=g_text)
