import argparse
import pickle

import numpy
import pandas
from keras.models import load_model
from keras.preprocessing import sequence

SEED = 7
numpy.random.seed(SEED)

emotion_labels = {0: 'love', 1: 'enthusiasm', 2: 'happiness', 3: 'fun', 4: 'relief', 5: 'surprise',
                  6: 'neutral', 7:  'empty', 8: 'boredom', 9: 'worry', 10: 'sadness', 11: 'anger',
                  12: 'hate'}


def analyze_tweets(model, dataset_path: str, tokenizer_path: str, max_words=37):
    dataset = pandas.read_csv(dataset_path, delimiter=' ', quotechar='|')
    tweets = dataset['Text']

    # Preprocess texts
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    preprocessed_tweets = tokenizer.texts_to_sequences(tweets)

    # Padding
    preprocessed_tweets = sequence.pad_sequences(preprocessed_tweets, maxlen=max_words)

    # Analyze
    predicted_classes = model.predict_classes(numpy.array(preprocessed_tweets), verbose=0)
    prediction_confidence = []
    for scores in model.predict(preprocessed_tweets, verbose=0):
        prediction_confidence.append(numpy.max(scores))
    results = zip(predicted_classes, prediction_confidence, tweets)

    # Calculate weighted results
    weighted_results = {'love': 0, 'enthusiasm': 0, 'happiness': 0, 'fun': 0, 'relief': 0, 'surprise': 0,
                  'neutral': 0, 'empty': 0, 'boredom': 0, 'worry': 0, 'sadness': 0, 'anger': 0,
                  'hate': 0}

    for result in results:
        weighted_results[emotion_labels[result[0]]] += 1 - (1 - result[1])

    weighted_results_sum = 0
    print('Weighted results:')
    for key in weighted_results:
        print('  - Emotion: {0}\n    weighted result: {1}'.format(key, weighted_results[key]))
        weighted_results_sum += weighted_results[key]

    print('Weighted results percentage:')
    for key in weighted_results:
        print('  - Emotion: {0}\n    weighted result percentage: {1}'.format(key,
                                                                     (weighted_results[key]/weighted_results_sum)*100))


def parse_args():
    parser = argparse.ArgumentParser(description='Perform emotion analysis on a given dataset of tweets.')
    parser.add_argument('-d', '--dataset-path', required=True,
                        type=str, help='Path to dataset CSV file.')
    parser.add_argument('-m', '--model-path', default='emotion_analysis_CNN.h5',
                        type=str, help='Path to a file with trained emotion analysis model.')
    parser.add_argument('-t', '--tokenizer-path', default='tokenizer.bin',
                        type=str, help='Path to a serialized keras.preprocessing.text.Tokenizer object'
                                       'used during model training.')
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_path)
    analyze_tweets(model, args.dataset_path, args.tokenizer_path)

if __name__ == '__main__':
    main()
