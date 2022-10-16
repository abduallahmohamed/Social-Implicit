import os
import pathlib
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import json


# =================================================================================================
# This script is used to convert the SDD data used by DAG net to the ETH/UCY dataformat.
# We already provide the converted data in ./datasets/sdd
# If you want to convert the data by yourself, first get the DAG version of SDD data from (https://github.com/alexmonti19/dagnet/blob/master/datasets/README.md).
# Then, set the variable DATASET_DIR to the sdd data location and run this script.
# =================================================================================================



DATASET_DIR = pathlib.Path('/path/to/SDD-DAG-Version/')
SDD_RAW_DIR = DATASET_DIR / 'all_data'
SDD_NPY_DIR = DATASET_DIR / 'sdd_npy'

save_root = 'datasets/sdd/'

def split():
    os.makedirs(os.path.join(save_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'test'), exist_ok=True)

    stats_file_full_path = DATASET_DIR.absolute() / 'stats.txt'
    stats_file = open(stats_file_full_path, 'w+')
    scenes_limits = {}

    all_files = [file_path for file_path in SDD_RAW_DIR.iterdir()]

    file_data = []
    for file in tqdm(all_files, desc='Reading files'):
        dname = file.name.split(".")[0]
        with open(file, 'r') as f:
            lines = []
            for line in f:
                line = line.strip().split(" ")
                line = [int(line[0]), int(line[1]), float(line[2]), float(line[3])]
                lines.append(np.asarray(line))
            file_data.append((dname, np.stack(lines)))

    training_split, test_split, validation_split, all_data = [], [], [], []
    for file in tqdm(file_data, desc='Preprocessing files'):
        scene_name = file[0]
        data = file[1]
        data_per_file = []
        # Frame IDs of the frames in the current dataset
        frame_list = np.unique(data[:, 0]).tolist()

        min_x, max_x = data[:, 2].min().item(), data[:, 2].max().item()
        min_y, max_y = data[:, 3].min().item(), data[:, 3].max().item()

        range_x = np.round(np.abs(max_x - min_x), decimals=3)
        range_y = np.round(np.abs(max_y - min_y), decimals=3)

        stats_file.write('SCENE {}:\nmin_x: {}, max_x: {}\nmin_y: {}, max_y: {}\nrange_x: {}, range_y: {}\n\n\n'.
                         format(scene_name, min_x, max_x, min_y, max_y, range_x, range_y))

        # round down min(s) to 1 decimal place
        min_x = math.floor(min_x * 10) / 10.0
        min_y = math.floor(min_y * 10) / 10.0

        # round up max(s) to 1 decimal place
        max_x = math.ceil(max_x * 10) / 10.0
        max_y = math.ceil(max_y * 10) / 10.0

        scenes_limits[scene_name] = {'x_min': min_x, 'x_max': max_x, 'y_min': min_y, 'y_max': max_y}

        for frame in frame_list:
            # Extract all pedestrians in current frame
            ped_in_frame = data[data[:, 0] == frame, :]
            data_per_file.append(ped_in_frame)

        # all_data.append((scene_name, np.concatenate(data_per_file)))

        n_trjs = len(data_per_file)

        # 70% --> training
        # training_split.append((scene_name, np.concatenate(data_per_file[: math.ceil(n_trjs * 0.7)])))
        train_df = pd.DataFrame(np.concatenate(data_per_file[:math.ceil(n_trjs*0.7)]),
                                columns=['frame_id', 'pedestrian_id', 'x', 'y'])

        # 10% --> validation
        # 20% --> test
        validation_test_split = data_per_file[math.ceil(n_trjs*0.7):]
        # validation_split.append((scene_name, np.concatenate(validation_test_split[:math.ceil(n_trjs * 0.1)])))
        val_df = pd.DataFrame(np.concatenate(validation_test_split[:math.ceil(n_trjs * 0.1)]),
                              columns=['frame_id', 'pedestrian_id', 'x', 'y'])
        # test_split.append((scene_name, np.concatenate(validation_test_split[math.ceil(n_trjs*0.1):])))
        test_df = pd.DataFrame(np.concatenate(validation_test_split[math.ceil(n_trjs * 0.1):]),
                               columns=['frame_id', 'pedestrian_id', 'x', 'y'])

        train_df.to_csv(os.path.join(save_root, 'train', '{}.txt'.format(scene_name)),
                        sep='\t', header=False, index=False)
        val_df.to_csv(os.path.join(save_root, 'val', '{}.txt'.format(scene_name)),
                      sep='\t', header=False, index=False)
        test_df.to_csv(os.path.join(save_root, 'test', '{}.txt'.format(scene_name)),
                       sep='\t', header=False, index=False)



    print('Splitting...')
    SDD_NPY_DIR.mkdir(exist_ok=True, parents=True)
    print('Saving splits...')
    np.save('{}/{}'.format(SDD_NPY_DIR, 'all_data.npy'), np.asarray(all_data))
    np.save('{}/{}'.format(SDD_NPY_DIR, 'train.npy'), np.asarray(training_split))
    np.save('{}/{}'.format(SDD_NPY_DIR, 'test.npy'), np.asarray(test_split))
    np.save('{}/{}'.format(SDD_NPY_DIR, 'validation.npy'), np.asarray(validation_split))

    print('Saving dataset stats...')
    stats_file.close()
    limits_json_full_path = DATASET_DIR.absolute() / 'limits.json'
    with open(limits_json_full_path, 'w+') as outfile:
        json.dump(scenes_limits, outfile, indent=2)


def main():
    split()
    print('All done.')


if __name__ == '__main__':
    main()
