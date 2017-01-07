import tweepy
import configparser
from models.tweet import PreprocessedTweet


class CollectorStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)
        preproc_tweet = PreprocessedTweet(text=status.text, timestamp=status.timestamp_ms,
                                          retweet_count=status.retweet_count)
        preproc_tweet.save_in_csv('tweets.csv')


def load_configuration(path='config.ini'):
    configuration = configparser.ConfigParser()
    configuration.read(path)
    return configuration


def initialize_twitter_api(twitter_credentials):
    auth = tweepy.OAuthHandler(twitter_credentials['consumer_key'], twitter_credentials['consumer_secret'])
    auth.set_access_token(twitter_credentials['access_token'], twitter_credentials['access_token_secret'])

    return tweepy.API(auth)


def collect_tweets(api):
    collector_stream_listener = CollectorStreamListener()
    collector_stream = tweepy.Stream(auth=api.auth, listener=collector_stream_listener)
    collector_stream.filter(track=['trump'])


if __name__ == '__main__':
    config = load_configuration()
    twitter_api = initialize_twitter_api(config['twitter_credentials'])
    collect_tweets(twitter_api)
