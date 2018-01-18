import argparse
from typing import List

import pandas


def load_dataset(path: str) -> pandas.DataFrame:
    dataset = pandas.read_csv(path, encoding='utf-8', sep=' ', quotechar='|')
    return dataset


def map_labels_to_ids(labels: List[str]) -> dict:
    """
    :param labels: List of labels.
    :return: A two way mapping of labels and ids.
    """
    id_label_tuples = [(id, label) for id, label in enumerate(labels)]
    label_id_tuples = [(label, id) for id, label in enumerate(labels)]
    return {k: v for k, v in id_label_tuples + label_id_tuples}


def classify_loop(dataset: pandas.DataFrame, text_column: str,
                  labels: List[str], result_dataset_path: str = 'classified-4.csv'):
    texts = dataset[text_column]
    if 'label' not in dataset.columns:
        dataset['label'] = pandas.Series([None for _ in range(len(dataset))], index=dataset.index)
    text_labels = dataset['label']
    label_id_mapping = map_labels_to_ids(labels)
    for i, text, label in zip((i for i in range(len(texts))), texts, text_labels):
        # Skip classification for already labelled data
        if label in labels:
            continue
        print(text)
        print('Classify (press I to skip, press any different key to stop):')
        for label in labels:
            print('{} - {}'.format(label, label_id_mapping[label]))
        try:
            selected = input('Action: ')
            if selected == 'I' or selected == 'i':
                continue
            classification = int(selected)
            text_label = label_id_mapping[classification]
            dataset.set_value(i, 'label', text_label)
        except ValueError as e:
            print(e)
            break

    dataset.to_csv(path_or_buf=result_dataset_path, encoding='utf-8', sep=' ', quotechar='|')


def parse_args():
    parser = argparse.ArgumentParser(description='Tool for manual classification of text from given dataset.')
    parser.add_argument('-d', '--dataset-path', required=True,
                        type=str, help='Path to dataset CSV file.')
    parser.add_argument('-r', '--result-path', required=True,
                        type=str, help='Path where result CSV file will be saved.')
    parser.add_argument('-t', '--text-column', required=True,
                        type=str, help='Name of text column in dataset that will be labelled.')
    parser.add_argument('-l', '--labels', nargs='+', required=True,
                        type=str, help='List of possible labels.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = load_dataset(path=args.dataset_path)
    classify_loop(dataset=dataset, result_dataset_path=args.result_path,
                  text_column=args.text_column,
                  labels=args.labels)
