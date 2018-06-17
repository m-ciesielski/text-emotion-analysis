import argparse
import os

import nltk
import nltk.sentiment.vader
import pandas


def is_tweet_negative(analysed_tweet: dict) -> bool:
    return analysed_tweet['compound'] <= -0.05


def is_tweet_positive(analysed_tweet: dict) -> bool:
    return analysed_tweet['compound'] >= 0.05


def classify_tweet(analyzed_tweet: dict) -> str:
    if is_tweet_positive(analyzed_tweet):
        return 'positive'
    elif is_tweet_negative(analyzed_tweet):
        return 'negative'
    else:
        return 'neutral'


def predict_sentiment(texts):
    sentiment_analyzer = nltk.sentiment.vader.SentimentIntensityAnalyzer()

    analyzed_texts = [sentiment_analyzer.polarity_scores(str(text)) for text in texts]
    sentiments = [classify_tweet(t) for t in analyzed_texts]

    return sentiments


def download_required_corpora():
    nltk.download(info_or_id="vader_lexicon")


def parse_args():
    parser = argparse.ArgumentParser(description='Perform sentiment analysis on given dataset of tweets.')
    parser.add_argument('-d', '--dataset-path', default=os.environ.get('DATASET_PATH'),
                        type=str, help='Path to dataset CSV file.')
    parser.add_argument('-o', '--output-path', default=os.environ.get('OUTPUT_PATH'),
                        type=str, help='Path to output CSV file.')
    return parser.parse_args()


def main():
    args = parse_args()
    download_required_corpora()

    dataset = pandas.read_csv(args.dataset_path, sep=' ', quotechar='|', encoding='utf-8')

    predicted_sentiment = predict_sentiment(dataset['Text'])
    dataset['sentiment'] = predicted_sentiment

    dataset.to_csv(args.output_path, sep=' ', quotechar='|', encoding='utf-8')


if __name__ == '__main__':
    main()
