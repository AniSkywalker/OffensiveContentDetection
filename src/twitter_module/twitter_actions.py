import codecs
import sys
from collections import defaultdict
import time
import tweepy
import os
import re
import urllib
sys.path.append('../../../')

# Consumer keys and access tokens, used for OAuth
from tweepy.error import TweepError

from OffensiveContentDetection.src.offensive_content_detection_model_CNN_LSTM_DNN import test_model

consumer_key = "1xd5E01MLPzcr0AN6Xm0iXyS3"
consumer_secret = "Z73tLVF1fBojSj1joZki1kEdzMrpjjXGh8wGJ4MKMnolkd8L3c"

access_token = "2807844465-rPINnrjMi3aonlgWQqAVBUAGSrPmigiVEwY2Lqx"
access_token_secret = "g0OBiXmXcl63ZFew6rByp4rcVNegKPptMzYuNYrvtyvOR"

reply_file = None

class twitter_api():
    _api = None
    _auth = None

    def __init__(self):
        self.__authenticate()

    def __authenticate(self):
        self._auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self._auth.set_access_token(access_token, access_token_secret)

    def get_user_timeline(self, username):
        return self._api.user_timeline(username, count=100)

    def get_exists_user(self, username):
        try:
            return self._api.lookup_users(screen_names=username)
        except(TweepError):
            return None

    def get_follower_ids(self, id=''):
        try:
            return self._api.followers_ids(id)['ids']
        except(TweepError):
            self.__authenticate()
            return []

    def get_user(self, id=''):
        return self._api.get_user(id)

    def get_user_screen_name(self, screen_name=''):
        return self._api.get_user(screen_name = screen_name)

    def get_status_details(self, id):
        return self._api.get_status(id)

    def get_friends_id(self, id):
        return self._api.friends_ids(id)['ids']

    def get_status(self, id):
        try:
            return self._api.get_status(id)['text']
        except tweepy.TweepError as e:
            if (e.args[0][0]['code'] == '144'):
                return e.message[0]['message']
            return None

    def get_all_tweets(self, screen_name,max_len=18000):
        # initialize a list to hold all the tweepy Tweets
        alltweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = self._api.user_timeline(screen_name=screen_name, count=200)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = alltweets[-1]['id'] - 1

        # print(alltweets[-1])

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0 and len(alltweets)<max_len:
            print("getting tweets before %s" % (oldest))

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = self._api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

            # save most recent tweets
            alltweets.extend(new_tweets)

            # update the id of the oldest tweet less one
            oldest = alltweets[-1]['id'] - 1

            print("...%s tweets downloaded so far" % (len(alltweets)))

        # transform the tweepy tweets into a 2D array that will populate the csv

        return  alltweets

    def get_all_liked_shared_tweets(self, screen_name):
        # initialize a list to hold all the tweepy Tweets
        alltweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = self._api.user_timeline(screen_name=screen_name, count=200)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = alltweets[-1]['id'] - 1

        # print(alltweets[-1])

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            print("getting tweets before %s" % (oldest))

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = self._api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

            # save most recent tweets
            alltweets.extend(new_tweets)

            # update the id of the oldest tweet less one
            oldest = alltweets[-1]['id'] - 1

            print("...%s tweets downloaded so far" % (len(alltweets)))

        # transform the tweepy tweets into a 2D array that will populate the csv

        outtweets = []
        for tweet in alltweets:
            if(int(tweet['favorite_count'])> 0 or int(tweet['retweet_count'])> 0):
                try:
                    outtweets.append([tweet['id_str'],tweet['favorite_count'],tweet['retweet_count'],tweet['quoted_status_id'], convert_one_line(tweet['text'])])
                except:
                    pass

        list_of_tweets = defaultdict()

        for outtweet in outtweets:
            try:
                tweet = self.get_status_details(id=outtweet[3])
                context_tweet = 'NA'
                if(tweet['in_reply_to_status_id_str']!=None):
                    tweet = self.ta.get_status_details(id=tweet['in_reply_to_status_id_str'])
                    context_tweet = tweet['text']

                print(tweet['text'])
                list_of_tweets[convert_one_line(tweet['text'].strip())]=[outtweet[0],outtweet[1],outtweet[2], context_tweet]
            except:
                pass
            time.sleep(9)

        return list_of_tweets


    def search_query(self, query):
        return self._api.search(q=query, count=200)

    def get_all_search_queries(self,query,max_len=18000):
        # initialize a list to hold all the tweepy Tweets
        alltweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = self._api.search(q=query, count=200)

        # save most recent tweets
        alltweets.extend(new_tweets['statuses'])

        # save the id of the oldest tweet less one
        oldest = alltweets[-1]['id'] - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets['statuses']) > 0 and len(alltweets)< max_len:
            print("getting tweets before %s" % (oldest))

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = self._api.search(q=query, count=200, max_id=oldest)

            # save most recent tweets
            alltweets.extend(new_tweets['statuses'])

            # update the id of the oldest tweet less one
            oldest = alltweets[-1]['id'] - 1

            print("...%s tweets downloaded so far" % (len(alltweets)))

        return alltweets

    def get_relies(self,res):
        print(len(res))

        tweets = defaultdict(int)

        with open(reply_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                token = line.strip().split('\t')
                tweets[token[1]+'\t'+token[2]+'\t'+token[3]] = token[0]

        print(len(tweets.keys()))

        with open(reply_file,'w') as fw:
            for r in res:
                try:
                    print(r['text'])
                    tweet = self.get_status_details(r['in_reply_to_status_id'])
                    reply_text = convert_one_line(tweet['text'])

                    favorite_count = tweet['favorite_count']

                    tweet = self.get_status_details(id=tweet['quoted_status_id_str'])

                    tweets[convert_one_line(tweet['text']) +'\t'+ reply_text + '\t' + convert_one_line(r['text'])] = favorite_count
                except:
                    pass
                time.sleep(6)

            for key,value in tweets.iteritems():
                fw.write(str(value) + '\t' + key + '\n')


def convert_one_line(text):
    token = re.split(' |\n|\r', text)
    return ' '.join([t.strip() for t in token])

class CustomStreamListener(tweepy.StreamListener):
    text = None

    def on_status(self, status):
        self.text = status.text

        if (status.truncated == True):
            print('prev::', self.text)
            self.text = status.extended_tweet['full_text']
            print('after::', self.text)




    def on_error(self, status_code):
        print(status_code)
        return True

class Interaction():
    ta = None
    timeline = None
    direct_tweets = None

    basepath = os.getcwd()[:os.getcwd().rfind('/')]

    test_file = basepath + '/../resource/test/test_offensive.txt'
    word_file_path = basepath + '/../resource/word_list.txt'

    output_file_offensive = basepath + '/../resource/text_model/TestResults_offensive.txt'
    model_file = basepath + '/../resource/text_model/weights/'
    vocab_file_path = basepath + '/../resource/text_model/vocab_list.txt'


    # hate_speech

    test_file = basepath + '/../resource/test/test_hate.txt'
    word_file_path = basepath + '/../resource/word_list.txt'

    output_file_hate = basepath + '/../resource/text_model/TestResults_hate.txt'
    model_file = basepath + '/../resource/text_model/weights/'
    vocab_file_path = basepath + '/../resource/text_model/vocab_list.txt'


    # emotion

    test_file = basepath + '/../resource/test/test_emotion.txt'
    word_file_path = basepath + '/../resource/word_list.txt'

    output_file_emotion = basepath + '/../resource/text_model/TestResults_emotion.txt'
    model_file = basepath + '/../resource/text_model/weights/'
    vocab_file_path = basepath + '/../resource/text_model/vocab_list.txt'


    # t_offensive = test_model(word_file_path, model_file, vocab_file_path, output_file)
    # t.load_trained_model()
    # t.predict(test_file)



    def __init__(self):
        self.ta = twitter_api()
        self.ta._api = tweepy.API(self.ta._auth, parser=tweepy.parsers.JSONParser())

        # self.t_offensive = test_model(self.word_file_path, self.model_file, self.vocab_file_path, self.output_file_offensive)
        # self.t_offensive.load_trained_model(model_file_name = 'offensive.json', weight_file='offensive.json.hdf5')
        #
        # self.t_hate = test_model(self.word_file_path, self.model_file, self.vocab_file_path, self.output_file_hate)
        # self.t_hate.load_trained_model(model_file_name = 'hate_speech.json', weight_file='hate_speech.json.hdf5')

        self.t_emotion = test_model(self.word_file_path, self.model_file, self.vocab_file_path, self.output_file_emotion)
        self.t_emotion.load_trained_model(model_file_name = 'emotion.json', weight_file='emotion.json.hdf5')


    def get_recent_tweets(self,screen_name):
        self.timelines = self.ta.get_all_tweets(screen_name, max_len=100)
        fw = open(self.output_file_offensive,'w')
        for tweet in self.timeline:
            fw.write('ID'+'-1'+'\t'+tweet['text'].strip()+'\n')
        fw.close()

        self.t_emotion.predict(self.output_file_offensive)








    def get_direct_tweets(self, screen_name):
        self.direct_tweets = self.ta.get_all_search_queries(screen_name, max_len=100)

    def get_audience_moods(self):
        audiences = [tweet['user']['screen_name'] for tweet in self.direct_tweets]
        audiences = audiences[:10]
        for audience in audiences:
            audience_timeline = ta.get_all_tweets(audience, max_len=200)






if __name__=='__main__':

    # ta = twitter_api()
    # ta._api = tweepy.API(ta._auth, parser=tweepy.parsers.JSONParser())
    #
    # tweets = ta.get_all_search_queries('@realDonaldTrump', max_len=100)

    inte = Interaction()
    inte.get_recent_tweets('realDonaldTrump')




    # ta.get_all_tweets('trumpscuttlebot')


    # print(ta.get_status_details(426071803560075264))
    #
    # handle ='HillaryClinton'
    # user =  ta.get_user_screen_name(handle)
    # print(user)
    #
    # image_url = str(user['profile_image_url'].replace('_normal.jpg','.jpg'))
    # urllib.urlretrieve(image_url, '/root/PycharmProjects/Projects/aclSarcasm/image/' + handle + '.png')

    # ta.get_all_tweets('TIME')



