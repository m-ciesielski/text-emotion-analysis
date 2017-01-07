import argparse
import configparser

import tweepy
from requests.packages.urllib3.exceptions import ProtocolError

from models.preprocessed_tweet import PreprocessedTweet


class CollectorStreamListener(tweepy.StreamListener):
    def __init__(self, csv_file_name):
        tweepy.StreamListener.__init__(self)
        self.csv_file_name = csv_file_name

    def on_status(self, status):
        print(status.text)
        preproc_tweet = PreprocessedTweet(text=status.text, timestamp_ms=status.timestamp_ms,
                                          hashtags=status.entities['hashtags'],
                                          retweet_count=status.retweet_count)
        preproc_tweet.save_in_csv(self.csv_file_name)


def load_configuration(path):
    configuration = configparser.ConfigParser()
    configuration.read(path)
    return configuration


def initialize_twitter_api(twitter_credentials):
    auth = tweepy.OAuthHandler(twitter_credentials['consumer_key'], twitter_credentials['consumer_secret'])
    auth.set_access_token(twitter_credentials['access_token'], twitter_credentials['access_token_secret'])

    return tweepy.API(auth)


def collect_tweets(api, keywords, csv_file_name):
    collector_stream_listener = CollectorStreamListener(csv_file_name)
    collector_stream = tweepy.Stream(auth=api.auth, listener=collector_stream_listener)
    while True:
        try:
            collector_stream.filter(track=keywords)
        except ProtocolError:
            continue
        except KeyboardInterrupt:
            collector_stream.disconnect()
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Collect tweets with given keyword and save them in CSV file.')
    parser.add_argument('-k', '--keywords', required=True,
                        metavar='K', type=str, nargs='+', help='List of tweets keywords.')
    parser.add_argument('-f', '--csv-file-name', required=True,
                        type=str, help='Name of CSV file where tweets will be saved.')
    parser.add_argument('-c', '--config-file', default='config.ini',
                        type=str, help='Path to configuration file.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_configuration(args.config_file)
    twitter_api = initialize_twitter_api(config['twitter_credentials'])
    collect_tweets(twitter_api, args.keywords, args.csv_file_name)

