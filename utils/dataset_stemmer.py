import argparse
from functools import partial
import multiprocessing

import nltk
import pandas


def stem_text(stemmer: nltk.stem.api.StemmerI, text: str):
    try:
        print('Stemming: {}'.format(text))
        stemmed_text = ' '.join(stemmer.stem(word) for word in text.split(' '))
        print(stemmed_text)
        return stemmed_text
    except IndexError as e:
        print('Unable to stem text: {}. Reason: {}'.format(text, e))
        return text


def parse_args():
    parser = argparse.ArgumentParser(description='Perform stemming using Porter stemmer on given dataset.')
    parser.add_argument('-d', '--dataset-path', required=True,
                        type=str, help='Path to dataset file.')
    parser.add_argument('-r', '--result-path', required=True,
                        type=str, help='Path to where stemmed dataset file will be produced.')
    parser.add_argument('-f', '--file-type', required=False, default='csv', type=str,
                        help='Dataset file type, either csv or text (one word for each line).')
    parser.add_argument('-c', '--column-name', required=False,
                        type=str, help='Name of the column on which stemming will be performed.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    nltk.download(info_or_id="treebank")
    stemmer = nltk.stem.PorterStemmer()
    if args.file_type == 'csv':
        dataset = pandas.read_csv(args.dataset_path)
        text_data = dataset[args.column_name]
    elif args.file_type == 'text':
        with open(args.dataset_path, mode='r', encoding='utf-8') as dataset_file:
            dataset = [text.split(' ') for text in dataset_file.readlines()]
        text_data = [item[0] for item in dataset]
    else:
        raise ValueError('Invalid dataset file type passed.')

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    stemmed_text_data = pool.map(partial(stem_text, stemmer), text_data)

    if args.file_type == 'csv':
        dataset[args.column_name] = stemmed_text_data
        dataset.to_csv(args.result_path, index=False, encoding='utf-8')
    elif args.file_type == 'text':
        for stemmed_text, dataset_item in zip(stemmed_text_data, dataset):
            dataset_item[0] = stemmed_text
        with open(args.result_path, mode='w', encoding='utf-8') as result_file:
            for item in dataset:
                result_file.write(' '.join(w for w in item))
