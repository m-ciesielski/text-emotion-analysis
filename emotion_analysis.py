import numpy
import pandas
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# Source: https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


# Source: https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list)-ngram_range+1):
            for ngram_value in range(2, ngram_range+1):
                ngram = tuple(new_list[i:i+ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# fix random seed for reproducibility
SEED = 7
DATASET_PATH = 'text_emotion.csv'
ngram_range = 2
max_features = 50000
nb_epoch = 5

numpy.random.seed(SEED)

# load the dataset
dataset = pandas.read_csv(DATASET_PATH)

tweets = dataset['content']
emotions = dataset['sentiment']

# Preprocessing
# Tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

preprocessed_texts = tokenizer.texts_to_sequences(tweets)

# Emotions
emotions_tokens = {'love': 0, 'enthusiasm': 1, 'happiness': 2, 'fun': 3, 'relief': 4, 'surprise': 5,
                   'neutral': 6, 'empty': 7, 'boredom': 8, 'worry': 9, 'sadness': 10, 'anger': 11,
                   'hate': 12}
emotions = [emotions_tokens[e] for e in emotions]
emotions = to_categorical(emotions)


if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in preprocessed_texts:
        for i in range(2, ngram_range+1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k+start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = numpy.max(list(indice_token.keys())) + 1

    # Augmenting preprocesed_texts with n-grams features
    preprocessed_texts = add_ngram(preprocessed_texts, token_indice, ngram_range)

# Padding
max_words = 37
preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=max_words)


# create the model
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32, input_length=max_words))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(13, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(preprocessed_texts, emotions, validation_split=0.2, nb_epoch=10, batch_size=256, verbose=1)

# Final evaluation of the model
scores = model.evaluate(preprocessed_texts, emotions, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Save model
model.save('emotion_analysis_CNN.h5')
