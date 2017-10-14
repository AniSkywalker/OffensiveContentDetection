import codecs
from collections import defaultdict
import time
import tweepy
import codecs
import re
import urllib

# Consumer keys and access tokens, used for OAuth
from tweepy.error import TweepError

consumer_key = "8bYbTwXwmeg7dE9Y9zzQI55bG"
consumer_secret = "kE9XPKStigeV3KReukSY2iFZW2oLVSrCv0m2Op77c2t3NryHUg"

access_token = "256094350-8EwRZcvtQPkqkMN8ks3MFWgYVdUSYaAQ5ePOHn8W"
access_token_secret = "S6Br3zTvoM6JxVgSMXsa9iEn5lSfAK54CyVaiJnyckOLK"

reply_file = '/home/PycharmProjects/Projects/twitter_browser/reply_file.txt'

love_tags = ['#adoration', '#affection', '#love', '#fondness', '#liking', '#lovin', '#loving']
joy_tags = ['#bliss', '#blithe', '#cheerfulness', '#gaiety', '#jolliness',
            '#joviality', '#joy', '#delight', '#enjoyment', '#gladness', '#happiness',
            '#jubilation', '#elation', '#euphoria', '#cheerful', '#gleeful', '#gayety',
            '#jolly', '#joyful', '#delightful', '#glad', '#happy', '#jubilant',
            '#elated',
            '#enthusiasm', '#zeal', '#zest', '#excitement', '#thrill', '#exciting',
            '#excited', '#thrilled', '#pride', '#proud', '#pridefulness', '#prideful',
            '#optimism', '#optimistic', '#hopefulness', '#hopeful', '#hoping',
            '#expecting',
            '#Relief', '#ease', '#relaxation', '#alleviation']
anger_tags = ['#irritation',
              '#annoyance',
              '#irritating', '#irritated', '#annoying', '#annoyed', '#bothersome',
              '#irksome', '#disturbing', '#anger', '#rage', '#outrage', '#fury', '#wrath',
              '#angry', '#frenzy', '#enraged', '#infuriated', '#irate', '#ireful',
              '#offended', '#outraged', '#raging', '#wrathful', '#furious', '#disgust',
              '#disgusting', '#disgusted', '#frustration', '#frustrated', '#frustrating',
              '#frustrate', '#envy', '#jealousy', '#jealous', '#envying']
surprise_tags = ['#surprise', '#amazement', '#astonishment', '#astoundment', '#Surprised', '#Surprising',
                 '#astonished', '#astounded', '#amazed', '#unexpected']
sad_tags = ['#depression',
            '#despair',
            '#hopelessness', '#sadness', '#unhappiness', '#sorrow', '#sad', '#depressed',
            '#depressing', '#desperation', '#despairing', '#unhappy', '#cheerless',
            '#dejected',
            '#brokenhearted', '#heartbroken', '#heartbreak', '#agony', '#suffering',
            '#hurt',
            '#anguish', '#suffer', '#pain', '#torment', '#torture', '#misery', '#grief',
            '#woe',
            '#dismay', '#disappointment', '#disappoint', '#disappointed',
            '#disappointing',
            '#letdown', '#neglect', '#loneliness', '#aloneness', '#homesickness',
            '#neglected',
            '#neglecting', '#lonely', '#lonesome', '#friendless', '#homesick',
            '#embarrassment',
            '#embarrass', '#embarrassed', '#embarrassing', '#abashment', '#awkwardness',
            '#abashed', '#ashamed', '#regretting', '#remorseful', '#guilt', '#regret',
            '#remorse', '#guilty', '#regretful']
fear_tags = ['#fear', '#fright', '#horror',
             '#terror', '#scare', '#panic', '#scared', '#frightened', '#fearful',
             '#panicking', '#panicked', '#panicky', '#anxiety', '#nervousness',
             '#tenseness', '#tension', '#uneasiness', '#worry', '#anxious', '#nervous',
             '#tense', '#uneasy', '#worried', '#worrying']


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

    def get_all_search_queries(self, query):
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


def word_ratio(inputText):
    word_count = 0
    tags_count = 0
    mention_count = 0
    link_count = 0
    words = inputText.strip().split(' ')
    for word in words:
        if len(word) > 0:
            if word.startswith("http"):
                link_count += 1
            elif word[0] == "@":
                mention_count += 1
            elif word[0] == '#':
                tags_count += 1
            else:
                word_count += 1;

    if word_count > 5:
        word_ratio = float(word_count) / len(words)
        return word_ratio * 100.0
    else:
        return 0


def crawl_by_file(ta):
    parsed_tweets = set()
    with codecs.open('/root/Downloads/sarcasm_wsd/resource/tweet.SARCASM.all.text.TRAIN', 'r', 'utf-8') as f:
        parsed_tweets.update([line.split('\t')[1] for line in f.readlines()])

    with codecs.open('/root/Downloads/sarcasm_wsd/resource/tweet.SARCASM.all.id.TRAIN', 'r', 'utf-8') as f:
        with codecs.open('/root/Downloads/sarcasm_wsd/resource/tweet.SARCASM.all.text.TRAIN', 'a', 'utf-8') as fw:
            lines = f.readlines()
            for i, line in enumerate(lines):
                print(i, line)

                tag, id = line.strip().split('\t')
                id = id
                if (not parsed_tweets.__contains__(id)):
                    status = ta.get_status(id)
                    if (status != None):
                        fw.write(tag + '\t' + str(id) + '\t' + unicode(convert_one_line(status.strip())) + '\n')
                    # else:
                    #     fw.write(tag+'\t'+str(id)+'\t'+'not found'+'\n')
                    time.sleep(1)


def get_tweet_source_by_file(ta):
    lines = open('/root/PycharmProjects/Projects/aclSarcasm/Bamman_sarcasm.ids.txt', 'r').readlines()

    with open('/root/PycharmProjects/Projects/aclSarcasm/Bamman_sarcasm.text.txt', 'w') as fw:
        for line in lines:
            try:
                id, tag = line.strip().split('\t')

                print(id)
                status = ta.get_status_details(id=id)
                context = None
                if (status['in_reply_to_status_id_str'] != None):
                    context = ta.get_status_details(id=status['in_reply_to_status_id_str'])

                if (context != None):
                    fw.write(
                        convert_one_line(status['text']) + '\t' + convert_one_line(context['text']) + '\t' + tag + '\n')
                else:
                    fw.write(convert_one_line(status['text']) + '\t' + 'NA' + '\t' + tag + '\n')

                time.sleep(12)
            except:
                print('error', line)


def choose_emotion(text):
    for word in text.strip().split(' '):
        if love_tags.__contains__(word):
            print('love \t' + text)
            return 'love'
        elif sad_tags.__contains__(word):
            print('sad\t' + text)
            return 'sad'
        elif joy_tags.__contains__(word):
            print('joy\t' + text)
            return 'joy'
        elif fear_tags.__contains__(word):
            print('fear\t' + text)
            return 'fear'
        elif anger_tags.__contains__(word):
            print('anger\t' + text)
            return 'anger'
        elif surprise_tags.__contains__(word):
            print('surprise\t' + text)
            return 'surprise'


def write_file(text):
    emotion = choose_emotion(text)
    if emotion != None:
        f = open('emotion.txt', 'a+')
        f.write('ID' + '\t' + str(emotion) + '\t' + convert_one_line(text) + '\n')


class CustomStreamListener(tweepy.StreamListener):
    text = None

    # min 5 words excl link how many hash if theyre all hash there no point, 70/30


    def on_status(self, status):
        if not hasattr(self, 'retweet_status'):
            if not str(self.text).startswith('RT'):
                if (status.truncated == True):
                    self.text = status.extended_tweet['full_text']
                else:
                    self.text = status.text
                self.text = convert_one_line(self.text)
                if (word_ratio(self.text) > 70):
                    write_file(self.text)


def on_error(self, status_code):
    print(status_code)
    return True


if __name__ == '__main__':
    ta = twitter_api()
    ta._api = tweepy.API(ta._auth, parser=tweepy.parsers.JSONParser())
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

    # ta.get_relies(ta.get_all_search_queries("@onlinesarcasm"))


    cStreamListener = CustomStreamListener()
    stream = tweepy.Stream(auth=ta._auth, parser=tweepy.parsers.JSONParser(), listener=cStreamListener)
    stream.filter(languages=['en'], track=['#adoration', '#affection', '#love', '#fondness', '#liking', '#lovin',
                                           '#loving', '#bliss', '#blithe', '#cheerfulness', '#gaiety', '#jolliness',
                                           '#joviality', '#joy', '#delight', '#enjoyment', '#gladness', '#happiness',
                                           '#jubilation', '#elation', '#euphoria', '#cheerful', '#gleeful', '#gayety',
                                           '#jolly', '#joyful', '#delightful', '#glad', '#happy', '#jubilant',
                                           '#elated',
                                           '#enthusiasm', '#zeal', '#zest', '#excitement', '#thrill', '#exciting',
                                           '#excited', '#thrilled', '#pride', '#proud', '#pridefulness', '#prideful',
                                           '#optimism', '#optimistic', '#hopefulness', '#hopeful', '#hoping',
                                           '#expecting',
                                           '#Relief', '#ease', '#relaxation', '#alleviation', '#irritation',
                                           '#annoyance',
                                           '#irritating', '#irritated', '#annoying', '#annoyed', '#bothersome',
                                           '#irksome', '#disturbing', '#anger', '#rage', '#outrage', '#fury', '#wrath',
                                           '#angry', '#frenzy', '#enraged', '#infuriated', '#irate', '#ireful',
                                           '#offended', '#outraged', '#raging', '#wrathful', '#furious', '#disgust',
                                           '#disgusting', '#disgusted', '#frustration', '#frustrated', '#frustrating',
                                           '#frustrate', '#envy', '#jealousy', '#jealous', '#envying', '#surprise',
                                           '#amazement', '#astonishment', '#astoundment', '#Surprised', '#Surprising',
                                           '#astonished', '#astounded', '#amazed', '#unexpected', '#depression',
                                           '#despair',
                                           '#hopelessness', '#sadness', '#unhappiness', '#sorrow', '#sad', '#depressed',
                                           '#depressing', '#desperation', '#despairing', '#unhappy', '#cheerless',
                                           '#dejected',
                                           '#brokenhearted', '#heartbroken', '#heartbreak', '#agony', '#suffering',
                                           '#hurt',
                                           '#anguish', '#suffer', '#pain', '#torment', '#torture', '#misery', '#grief',
                                           '#woe',
                                           '#dismay', '#disappointment', '#disappoint', '#disappointed',
                                           '#disappointing',
                                           '#letdown', '#neglect', '#loneliness', '#aloneness', '#homesickness',
                                           '#neglected',
                                           '#neglecting', '#lonely', '#lonesome', '#friendless', '#homesick',
                                           '#embarrassment',
                                           '#embarrass', '#embarrassed', '#embarrassing', '#abashment', '#awkwardness',
                                           '#abashed', '#ashamed', '#regretting', '#remorseful', '#guilt', '#regret',
                                           '#remorse', '#guilty', '#regretful', '#fear', '#fright', '#horror',
                                           '#terror', '#scare', '#panic', '#scared', '#frightened', '#fearful',
                                           '#panicking', '#panicked', '#panicky', '#anxiety', '#nervousness',
                                           '#tenseness', '#tension', '#uneasiness', '#worry', '#anxious', '#nervous',
                                           '#tense', '#uneasy', '#worried', '#worrying'])

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
