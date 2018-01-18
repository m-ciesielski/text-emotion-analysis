import argparse

import pandas


DEFAULT_COLUMN_MAPPING = {'Text': 'content', 'label': 'sentiment'}


def load_additional_dataset(path: str) -> pandas.DataFrame:
    dataset = pandas.read_csv(path, encoding='utf-8', sep=' ', quotechar='|')
    dataset.dropna(inplace=True)
    print(len(dataset))
    return dataset


def load_base_dataset(path: str) -> pandas.DataFrame:
    dataset = pandas.read_csv(path)
    return dataset


def augment_dataset(base_dataset: pandas.DataFrame, additional_dataset: pandas.DataFrame,
                    column_mappings: dict) -> pandas.DataFrame:
    additional_dataset.rename(columns=column_mappings, inplace=True)
    for missing_column in set(base_dataset.columns) - set(additional_dataset.columns):
        additional_dataset[missing_column] = None

    valid_columns = base_dataset.columns
    augmented_dataset = base_dataset.append(additional_dataset)
    for invalid_column in set(augmented_dataset.columns) - set(valid_columns):
        augmented_dataset.drop(invalid_column, inplace=True, axis=1)
    return augmented_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Tool for manual classification of text from given dataset.')
    parser.add_argument('-d', '--dataset-path', required=True,
                        type=str, help='Path to base dataset CSV file.')
    parser.add_argument('-a', '--additional-dataset-path', required=True,
                        type=str, help='Path to additional dataset CSV file.')
    parser.add_argument('-r', '--result-path', required=True,
                        type=str, help='Path where result CSV file will be saved.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    addiitonal_dataset = load_additional_dataset(args.additional_dataset_path)
    base_dataset = load_base_dataset(args.dataset_path)
    augmented_dataset = augment_dataset(base_dataset, addiitonal_dataset, DEFAULT_COLUMN_MAPPING)
    augmented_dataset.to_csv(args.result_path, encoding='utf-8', quotechar='"', index=False)
