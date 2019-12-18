import os
import pandas as pd


DATA_DIR = os.path.join('.', 'data')
ORIGINAL_DATA = 'original.csv'

def build_data_dir():
    for dir_name in ['conventional', 'organic']:
        os.makedirs(os.path.join(DATA_DIR, dir_name), exist_ok=True)

def main():
    """
    Script to preprocess raw data into seperate files.
    Creates directories:
        ./data/conventional/
        ./data/organic/

    Writes one CSV file containing the AveragePrice data 
    per region, e.g. Boston, inside both the conventional 
    and organic directories. 
    """
    df = pd.read_csv(os.path.join(DATA_DIR, ORIGINAL_DATA), 
                     index_col='Date')

    build_data_dir()

    for type_ in df.type.unique():
        for region in df.region.unique():
            df.loc[(df.type == type_) & (df.region == region)] \
            .drop(filter(lambda col: col != 'AveragePrice', df.columns), axis=1) \
            .sort_index() \
            .to_csv(os.path.join(DATA_DIR, type_, f'{region}.csv'))

if __name__ == '__main__':
    main()
