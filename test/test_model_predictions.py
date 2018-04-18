import argparse
import pickle

import numpy
from keras.models import load_model
from keras.preprocessing import sequence
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

from training.train_cnn import EMOTION_VALUES_MAP, EMOTION_LABELS_MAP, MAX_WORDS, \
    top_2_categorical_accuracy, create_text_representation_vectors, create_glove_embedding_index

SEED = 7
numpy.random.seed(SEED)


def get_prediction_accuracy(prediction_array: numpy.array):
    return numpy.max(prediction_array)


def get_predicted_emotion(prediction_array: numpy.array):
    predicted_class_index = numpy.argmax(prediction_array)
    return EMOTION_VALUES_MAP[predicted_class_index]


def parse_args():
    parser = argparse.ArgumentParser(description='Perform emotion analysis on a given dataset.')
    parser.add_argument('-m', '--model-path', default='emotion_analysis_CNN_keras_2.h5',
                        type=str, help='Path to a file with trained emotion analysis model.')
    parser.add_argument('-t', '--tokenizer-path', default='tokenizer.pkl',
                        type=str, help='Path to a serialized keras.preprocessing.text.Tokenizer object'
                                       'used during model training.')
    parser.add_argument('-g', '--glove-embeddings-path', default='glove_embeddings/glove.6B.300d.txt',
                        type=str, help='Path to glove embeddings file.')
    parser.add_argument('-d', '--glove-embeddings-dim', default=300,
                        type=int, help='GloVe embeddings dimension.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load model
    model = load_model(args.model_path, custom_objects={'top_2_categorical_accuracy': top_2_categorical_accuracy})

    # Prepare test data
    test_tweets = ['Yay :D',
                   'I hate Trump and his supporters.',
                   'I hate you, but I love you.',
                   'I love you, but I hate you.',
                   'I just finished watching your Stanford iPhone Class session. I really appreciate it. You Rock!',
                   'Only one exam left, and i am so happy for it :D',
                   'Math review. Im going to fail the exam.',
                   "I've got math exam tomorrow.",
                   'Fuck no internet damn time warner!',
                   '@stellargirl I loooooooovvvvvveee my Kindle2. Not that the DX is cool, but the 2 is fantastic in its own right.',
                   'Trump is becoming president tomorrow. We voted for him as a country and we need to accept it',
                   'Trump is our president now and you have to respect him!',
                   'dammit! hulu desktop has totally screwed up my ability to talk to a particular port on one of our dev servers. so i can\'t watch and code',
                   'blablablalbla',
                   'It is so annoying when she starts typing on her computer in the middle of the night!',
                   'I hate Bakersfield and I hate the Ports, let me go home already.  I want to start my vacation.',
                   'I would love to meet with you tonight.',
                   'Yay, we have done this!',
                   'New law introduced by president is disgusting and shameful!',
                   'Wow, I was not expecting this!',
                   'TRUMP WILL WIN AMERICAN PEOPLE NOT GOIN TO PUT UPWITH THIS CRAP WE NEED TO CLEAN HOUSE IN DEMOCROOKED WASHINGTON',
                   'Donald Trump citing Julian Assange is an utter disgrace. Profoundly worrying about where and how PEOTUS gets his in…',
                   'Pence is incredibly dangerous. His views are horrific, but his mild-mannered style fools many',
                   'Trump plans to quickly issue executive orders on Obamacare -  woohoo!',
                   'After democrats fought for her trans rights. Pathetic bitch',
                   'RT @kicksb4rent_: After democrats fought for her trans rights. Pathetic bitch',
                   'Sr Trump,the intelligence repo is devastating.Losing election by more than 3M votes and in addition this.Are you a leg…',
                   'TRUMP,  when will you understand that I am not paying for that fucken wall. Be clear with US tax payers. They will pay f…',
                   'TRUMP,  when will you understand that I am not paying for that fucken wall. Be clear with US tax payers. They will pay f…']

    # Load dumped tokenizer
    with open(args.tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    preprocessed_test_tweets = tokenizer.texts_to_sequences(test_tweets)
    preprocessed_test_tweets = sequence.pad_sequences(preprocessed_test_tweets, maxlen=MAX_WORDS)

    embedding_index = create_glove_embedding_index(glove_embeddings_file_path=args.glove_embeddings_path,)
    trvs = create_text_representation_vectors(texts=preprocessed_test_tweets,
                                              word_index=tokenizer.word_index,
                                              embeddings_index=embedding_index,
                                              glove_embeddings_dim=args.glove_embeddings_dim)
    # Change trvs to three dimensional sequence of vectors
    trvs = numpy.array(trvs)
    trvs = numpy.expand_dims(trvs, axis=2)

    predictions = model.predict(trvs)

    explainer = LimeTextExplainer(class_names=['love', 'fun', 'neutral', 'worry', 'sadness',
                                  'hate'], kernel_width=12)

    def lime_predict(texts):
        preprocessed_texts = tokenizer.texts_to_sequences(texts)
        preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=MAX_WORDS)
        lime_trvs = create_text_representation_vectors(texts=preprocessed_texts,
                                                       word_index=tokenizer.word_index,
                                                       embeddings_index=embedding_index)
        # Change trvs to three dimensional sequence of vectors
        lime_trvs = numpy.array(lime_trvs)
        lime_trvs = numpy.expand_dims(lime_trvs, axis=2)
        return model.predict(lime_trvs, batch_size=256)

    for tweet, prediction in zip(test_tweets, predictions):
        print('Emotion: {}, Accuracy: {:.2f}, Text: {}'.format(get_predicted_emotion(prediction),
                                                               get_prediction_accuracy(prediction),
                                                               tweet))

    for tweet, prediction in zip(test_tweets, predictions):
        # predicted_label = numpy.argmax(prediction)
        exp = explainer.explain_instance(tweet, lime_predict, top_labels=2,
                                         num_features=MAX_WORDS)
        for label in exp.available_labels():
            fig = exp.as_pyplot_figure(label=label)
        plt.show()
