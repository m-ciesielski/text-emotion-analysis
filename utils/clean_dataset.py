import argparse
from multiprocessing import Pool

import pandas


def is_tweet_english(text: str):
    raise NotImplementedError


def is_tweet_referring_to_another(text: str):
    if 'https://t.co' in text:
        return True
    else:
        return False


def is_tweet_valid(tweet: (int, str)):
    print('Analyzing tweet #{0}'.format(tweet[0]))
    return is_tweet_english(tweet[1]) and not is_tweet_referring_to_another(tweet[1])


def clean_dataset(dataset_path: str, output_path: str):
    dataset = pandas.read_csv(dataset_path, delimiter=' ', quotechar='|')
    tweets = dataset['Text']

    pool = Pool()
    pool.map(is_tweet_valid, enumerate(tweets))


def parse_args():
    parser = argparse.ArgumentParser(description='Cleans given dataset from non-english tweets and'
                                                 'tweets that are referencing to another tweet.')
    parser.add_argument('-d', '--dataset-path', required=True,
                        type=str, help='Path to dataset CSV file.')
    parser.add_argument('-o', '--output-path', required=True,
                        type=str, help='Path to where a cleaned dataset will be written.')
    return parser.parse_args()


def main():
    args = parse_args()
    clean_dataset(args.dataset_path, args.output_path)

if __name__ == '__main__':
    main()
