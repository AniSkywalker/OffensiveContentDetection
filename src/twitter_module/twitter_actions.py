import codecs
from collections import defaultdict
import time
import tweepy
import codecs
import re
import urllib

# Consumer keys and access tokens, used for OAuth
from tweepy.error import TweepError

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

    def get_all_tweets(self, screen_name):
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

        outtweets = [[tweet['id_str'],tweet['favorite_count'],tweet['retweet_count'], convert_one_line(tweet['text']).encode("utf-8")] for tweet in alltweets]

        # write the csv
        with open('crawl/' + '%s_tweets.csv' % screen_name, 'wb') as f:
            for outtweet in outtweets:
                f.write(str(outtweet[0]) + '\t' + str(outtweet[1]) + '\t'+ str(outtweet[2]) + '\t' + outtweet[3].strip() + '\n')

        pass

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

    def get_all_search_queries(self,query):
        # initialize a list to hold all the tweepy Tweets
        alltweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = self._api.search(q=query, count=200)

        # save most recent tweets
        alltweets.extend(new_tweets['statuses'])

        # save the id of the oldest tweet less one
        oldest = alltweets[-1]['id'] - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets['statuses']) > 0:
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



if __name__=='__main__':

    ta = twitter_api()
    ta._api = tweepy.API(ta._auth, parser=tweepy.parsers.JSONParser())

    ta.get_all_search_queries('@realDonaldTrump')
    ta.get_all_tweets('trumpscuttlebot')


    # print(ta.get_status_details(426071803560075264))
    #
    # handle ='HillaryClinton'
    # user =  ta.get_user_screen_name(handle)
    # print(user)
    #
    # image_url = str(user['profile_image_url'].replace('_normal.jpg','.jpg'))
    # urllib.urlretrieve(image_url, '/root/PycharmProjects/Projects/aclSarcasm/image/' + handle + '.png')

    # ta.get_all_tweets('TIME')

    # ta.get_relies(ta.get_all_search_queries("@onlinesarcasm"))


    # cStreamListener = CustomStreamListener()
    # stream = tweepy.Stream(auth=ta._auth, parser=tweepy.parsers.JSONParser(), listener=cStreamListener)
    # stream.filter(languages=['en'],track=[' as '])

    # results = ta._api.search(q=' as ', count=100)

    # for result in results['statuses']:
    #
    #     text = result['text']
    #     if(not text.startswith('RT ')):
    #         print(text)





    # crawl_by_file(ta)
    # get_tweet_source_by_file(ta)







    # list=set()
    #
    #
    # list.update(str(x) for x in ta.get_friends_id(id='onlinesarcasm'))
    # print(len(list),list)


    # print(t['id_str'])
    # twitter_account_list = []
    # twitter_account_list.append(ta.api.me()['id'])
    #
    # i=0

    # while(i<len(twitter_account_list)):
    #     time.sleep(1)
    #     id = twitter_account_list[i]
    #     i+=1
    #     screen_name = ta.get_user(id)['screen_name']
    #     followers = ta.get_follower_ids(id)
    #     print(screen_name,len(followers))
    #     if(len(followers)>100):
    #         print(screen_name, len(followers))
    #     for fid in followers:
    #         if(not twitter_account_list.__contains__(fid)):
    #             twitter_account_list.append(fid)

