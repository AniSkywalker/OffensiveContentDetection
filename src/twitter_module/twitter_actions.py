import codecs
import sys

sys.path.append('../../')

from collections import defaultdict
import time
import tweepy
import re
import urllib

# Consumer keys and access tokens, used for OAuth
from tweepy.error import TweepError

from twitter_browser.AWC_browser.awc_browser import analyze_word_crawler

consumer_key = "1xd5E01MLPzcr0AN6Xm0iXyS3"
consumer_secret = "Z73tLVF1fBojSj1joZki1kEdzMrpjjXGh8wGJ4MKMnolkd8L3c"

access_token = "2807844465-rPINnrjMi3aonlgWQqAVBUAGSrPmigiVEwY2Lqx"
access_token_secret = "g0OBiXmXcl63ZFew6rByp4rcVNegKPptMzYuNYrvtyvOR"

reply_file = None
crawl_file = '/home/PycharmProjects/Projects/twitter_browser/resource/sarcasm_context_moods_v2.txt'
search_file = '/home/PycharmProjects/Projects/twitter_browser/resource/search_sarcasm_context_moods_v2.txt'

awc = analyze_word_crawler()


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
        return self._api.get_user(screen_name=screen_name)

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

    def get_all_search_with_context_awc(self,query,max_len=18000):
        # initialize a list to hold all the tweepy Tweets
        alltweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = self._api.search(q=query, count=200)

        # save most recent tweets
        alltweets.extend(new_tweets['statuses'])

        # save the id of the oldest tweet less one
        oldest = alltweets[-1]['id'] - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets['statuses']) > 0 and len(alltweets) < max_len:
            print("getting tweets before %s" % (oldest))

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = self._api.search(q=query, count=200, max_id=oldest)

            # save most recent tweets
            alltweets.extend(new_tweets['statuses'])

            # update the id of the oldest tweet less one
            oldest = alltweets[-1]['id'] - 1

            print("...%s tweets downloaded so far" % (len(alltweets)))


        data = []
        for tweet in alltweets:
            if(tweet['in_reply_to_status_id']!=None and tweet['truncated'] == False and tweet['lang']=='en'):
                screen_name = tweet['user']['screen_name']
                awc.crawl_by_twitter_handle(screen_name)
                print(str(tweet['text']),awc.get_dimensions_as_string(),str(tweet['in_reply_to_status_id']),str(screen_name))
                data.append([str(tweet['text']),awc.get_dimensions_as_string(),str(tweet['in_reply_to_status_id']),str(screen_name)])
                time.sleep(1)

        return data


    def get_all_tweets(self, screen_name, max_len=18000):
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
        while len(new_tweets) > 0 and len(alltweets) < max_len:
            print("getting tweets before %s" % (oldest))

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = self._api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

            # save most recent tweets
            alltweets.extend(new_tweets)

            # update the id of the oldest tweet less one
            oldest = alltweets[-1]['id'] - 1

            print("...%s tweets downloaded so far" % (len(alltweets)))

        # transform the tweepy tweets into a 2D array that will populate the csv

        outtweets = [[tweet['id_str'], tweet['favorite_count'], tweet['retweet_count'],
                      convert_one_line(tweet['text']).encode("utf-8")] for tweet in alltweets]

        # write the csv
        with open('crawl/' + '%s_tweets.csv' % screen_name, 'wb') as f:
            for outtweet in outtweets:
                f.write(str(outtweet[0]) + '\t' + str(outtweet[1]) + '\t' + str(outtweet[2]) + '\t' + outtweet[
                    3].strip() + '\n')

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
            if (int(tweet['favorite_count']) > 0 or int(tweet['retweet_count']) > 0):
                try:
                    outtweets.append(
                        [tweet['id_str'], tweet['favorite_count'], tweet['retweet_count'], tweet['quoted_status_id'],
                         convert_one_line(tweet['text'])])
                except:
                    pass

        list_of_tweets = defaultdict()

        for outtweet in outtweets:
            try:
                tweet = self.get_status_details(id=outtweet[3])
                context_tweet = 'NA'
                if (tweet['in_reply_to_status_id_str'] != None):
                    tweet = self.ta.get_status_details(id=tweet['in_reply_to_status_id_str'])
                    context_tweet = tweet['text']

                print(tweet['text'])
                list_of_tweets[convert_one_line(tweet['text'].strip())] = [outtweet[0], outtweet[1], outtweet[2],
                                                                           context_tweet]
            except:
                pass
            time.sleep(9)

        return list_of_tweets

    def search_query(self, query):
        return self._api.search(q=query, count=200)



    def get_all_search_queries(self, query, max_len=18000):
        # initialize a list to hold all the tweepy Tweets
        alltweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = self._api.search(q=query, count=200)

        # save most recent tweets
        alltweets.extend(new_tweets['statuses'])

        # save the id of the oldest tweet less one
        oldest = alltweets[-1]['id'] - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets['statuses']) > 0 and len(alltweets) < max_len:
            print("getting tweets before %s" % (oldest))

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = self._api.search(q=query, count=200, max_id=oldest)

            # save most recent tweets
            alltweets.extend(new_tweets['statuses'])

            # update the id of the oldest tweet less one
            oldest = alltweets[-1]['id'] - 1

            print("...%s tweets downloaded so far" % (len(alltweets)))

        return alltweets

    def get_relies(self, res):
        print(len(res))

        tweets = defaultdict(int)

        with open(reply_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                token = line.strip().split('\t')
                tweets[token[1] + '\t' + token[2] + '\t' + token[3]] = token[0]

        print(len(tweets.keys()))

        with open(reply_file, 'w') as fw:
            for r in res:
                try:
                    print(r['text'])
                    tweet = self.get_status_details(r['in_reply_to_status_id'])
                    reply_text = convert_one_line(tweet['text'])

                    favorite_count = tweet['favorite_count']

                    tweet = self.get_status_details(id=tweet['quoted_status_id_str'])

                    tweets[convert_one_line(tweet['text']) + '\t' + reply_text + '\t' + convert_one_line(
                        r['text'])] = favorite_count
                except:
                    pass
                time.sleep(6)

            for key, value in tweets.iteritems():
                fw.write(str(value) + '\t' + key + '\n')


def convert_one_line(text):
    token = re.split(' |\n|\r', text)
    return ' '.join([t.strip() for t in token])


class CustomStreamListener(tweepy.StreamListener):
    text = None
    screen_name = None

    def on_status(self, status):
        fw = open(crawl_file, 'a')
        self.text = status.text

        if (status.truncated == True):
            self.text = status.extended_tweet['full_text']


        if (status.in_reply_to_status_id_str != None):
            self.screen_name = status.user.screen_name
            awc.crawl_by_twitter_handle(self.screen_name)
            try:
                source = ta.get_status_details(int(status.in_reply_to_status_id_str))
                if (source['truncated'] == False):
                    self.context = source['text']
                    fw.write('TrainSen' + '\t' + '1' + '\t' + convert_one_line(str(self.text)) + '\t' + awc.get_dimensions_as_string() + '\t'
                             + convert_one_line(str(self.context)) + '\t' + convert_one_line(str(self.screen_name))+'\n')
                    print('TrainSen' + '\t' + '1' + '\t' + str(self.text) + '\t' + awc.get_dimensions_as_string() + '\t'
                             + str(self.context) + '\t' + str(self.screen_name))

            except:
                print('unavailable source')
                raise
            time.sleep(6)
        fw.close()

    def on_error(self, status_code):
        print(status_code)
        return True


class interaction():
    ta = twitter_api()
    ta._api = tweepy.API(ta._auth, parser=tweepy.parsers.JSONParser())

    def get_recent_tweets(self, query):
        tweets = ta.get_all_search_queries(query, max_len=1000)
        texts = [tweet['text'] for tweet in tweets if not tweet['text'].startswith('RT')]
        # fw =


if __name__ == '__main__':
    ta = twitter_api()
    ta._api = tweepy.API(ta._auth, parser=tweepy.parsers.JSONParser())

    # ta.get_all_search_queries('@realDonaldTrump')
    # ta.get_all_tweets('trumpscuttlebot')

    # tweet = ta.get_status_details(922951419149340672)
    # if(tweet['in_reply_to_status_id']==None):
    #     print(tweet['text'],tweet['lang'])
    #
    # tweet = ta.get_status_details(923011949763239936)
    # if(tweet['in_reply_to_status_id']!=None):
    #     print(tweet['text'])


    # fw = open(search_file, 'a')
    # data = ta.get_all_search_with_context_awc('#sarcasm')
    # for token in data:
    #     fw.write('TrainSen' + '\t' + '1' + '\t' + convert_one_line(token[0]) + '\t' + token[1] + '\t'
    #                          + convert_one_line(token[2]) + '\t' + convert_one_line(token[3])+'\n')
    #
    # fw.close()





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
    # stream.filter(languages=['en'], track=['#sarcasm'])

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
