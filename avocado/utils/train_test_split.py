import os
import glob


DATA_DIR = os.path.join('.', 'data')
conventional_dir = os.path.join(DATA_DIR, 'conventional')
organic_dir = os.path.join(DATA_DIR, 'organic')

TRAIN_PARTITION = 0.8
TEST_PARTITION = 0.2

def partition(data, split='train'):
    if split == 'train':
        return data[:int(len(data)*TRAIN_PARTITION)]
    else:
        return data[int(len(data)*TRAIN_PARTITION):]

def main():
    for dir_name in [conventional_dir, organic_dir]:
        os.makedirs(os.path.join(dir_name, 'train'), exist_ok=True)
        os.makedirs(os.path.join(dir_name, 'ground_truth'), exist_ok=True)


    for dir_name in [conventional_dir, organic_dir]:
        data_files = glob.glob(os.path.join(dir_name, 'raw', '*.csv'))

        for filename in data_files:
            with open(filename) as file:
                lines = file.readlines()

            header = lines[0]
            # remove header
            lines = lines[1:]

            # csv file name without path
            fname = filename.split(os.path.sep)[-1]
            
            for split in ['train', 'ground_truth']:
                with open(os.path.join(dir_name, split, fname) , 'w+') as file:
                    file.write(header)
                    file.write(''.join(partition(lines, split=split)))


if __name__ == '__main__':
    main()