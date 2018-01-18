import pickle

import numpy
from keras.models import load_model
from keras.preprocessing import sequence

from training.train_cnn import top_2_categorical_accuracy, emotion_indices

SEED = 7
numpy.random.seed(SEED)


def get_prediction_accuracy(prediction_array):
    return numpy.max(prediction_array)


def get_predicted_emotion(prediction_array):
    predicted_class_index = numpy.argmax(prediction_array)
    return emotion_indices[predicted_class_index]

ngram_range = 1
max_features = 50000

# load model
model = load_model('emotion_analysis_CNN_keras_2_new_2.h5', custom_objects={'top_2_categorical_accuracy': top_2_categorical_accuracy})

# Prepare test data
test_tweets = ['I hate Trump',
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
with open('tokenizer_new.bin', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)


preprocessed_test_tweets = tokenizer.texts_to_sequences(test_tweets)

# Padding
max_words = 37
preprocessed_test_tweets = sequence.pad_sequences(preprocessed_test_tweets, maxlen=max_words)

predictions = model.predict(numpy.array(preprocessed_test_tweets))

for i, p in enumerate(predictions):
    print('Emotion: {}, Accuracy: {:.2f}, Text: {}'.format(get_predicted_emotion(p), get_prediction_accuracy(p),
                                                           test_tweets[i]))