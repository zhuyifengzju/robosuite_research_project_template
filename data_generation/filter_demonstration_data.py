
import argparse
import h5py
import numpy as np

from robomimic.utils.file_utils import create_hdf5_filter_key

def create_demonstration_labels(hdf5_path, num_demo_count=[20, 30]):
    """
    Splits data into training set and validation set from HDF5 file.
    Args:
        hdf5_path (str): path to the hdf5 file
            to load the transitions from
        val_ratio (float): ratio of validation demonstrations to all demonstrations
        filter_key (str): if provided, split the subset of demonstration keys stored
            under mask/@filter_key instead of the full set of demonstrations
    """

    # retrieve number of demos
    f = h5py.File(hdf5_path, "r")

    # filter_key = []
    # for num_demo in num_demo_count:
    #     filter_key.append(f"{num_demo}_demos")
    filter_key = None
    if filter_key is not None:
        print("using filter key: {}".format(filter_key))
        demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])])
    else:
        demos = sorted(list(f["data"].keys()))
    num_demos = len(demos)
    f.close()

    # get random split
    num_demos = len(demos)
    for num_val in num_demo_count:
        mask = np.zeros(num_demos)
        mask[:num_val] = 1.
        np.random.shuffle(mask)
        mask = mask.astype(int)
        inds = mask.nonzero()[0]

        keys = [demos[i] for i in inds]
        key_name = f"num_demo_{num_val}"

        create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=keys, key_name=key_name)

    # # pass mask to generate split
    # name_1 = "train"
    # name_2 = "valid"
    # if filter_key is not None:
    #     name_1 = "{}_{}".format(filter_key, name_1)
    #     name_2 = "{}_{}".format(filter_key, name_2)

    # train_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=train_keys, key_name=name_1)
    # valid_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=valid_keys, key_name=name_2)

    # print("Total number of train samples: {}".format(np.sum(train_lengths)))
    # print("Average number of train samples {}".format(np.mean(train_lengths)))

    # print("Total number of valid samples: {}".format(np.sum(valid_lengths)))
    # print("Average number of valid samples {}".format(np.mean(valid_lengths)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    args = parser.parse_args()
    # seed to make sure results are consistent
    np.random.seed(0)
    create_demonstration_labels(args.dataset)
