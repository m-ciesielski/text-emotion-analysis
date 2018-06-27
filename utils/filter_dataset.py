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


def clean_dataset_sentiment(dataset_path: str, output_path: str, sentiment_to_keep: str, emotion: str):
    dataset = pandas.read_csv(dataset_path, sep=' ', quotechar='|', encoding='cp1250')
    dataset.drop_duplicates(subset='Text', inplace=True)
    dataset.drop('Unnamed: 0', axis=1, inplace=True)
    sentiments = dataset['sentiment']

    indices_of_texts_to_remove = []
    for i, text_sentiment in enumerate(sentiments):
        if text_sentiment != sentiment_to_keep:
            indices_of_texts_to_remove.append(i)

    dataset.drop(dataset.index[indices_of_texts_to_remove], inplace=True)

    dataset.drop('sentiment', axis=1, inplace=True)
    dataset.drop('Hashtags', axis=1, inplace=True)
    dataset.drop('RetweetCount', axis=1, inplace=True)
    dataset.drop('Timestamp', axis=1, inplace=True)
    dataset.rename(index=str, columns={"Text": "content"}, inplace=True)

    dataset['sentiment'] = emotion
    dataset['author'] = 'unknown'
    dataset['tweet_id'] = 'unknown'

    dataset.to_csv(output_path, columns=['author', 'content', 'sentiment', 'tweet_id'],
                   encoding='utf-8', index=False)


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
    clean_dataset_sentiment(args.dataset_path, args.output_path, sentiment_to_keep='negative', emotion='hate')

if __name__ == '__main__':
    main()
