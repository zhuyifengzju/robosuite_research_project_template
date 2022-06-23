from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

def get_dataset(dataset_path, obs_modality, seq_len=1, filter_key=None, hdf5_cache_mode="low_dim", *args, **kwargs):
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            all_obs_keys=all_obs_keys,
            verbose=True)

    seq_len = seq_len
    filter_key = filter_key

    dataset = SequenceDataset(
                hdf5_path=dataset_path,
                obs_keys=shape_meta["all_obs_keys"],
                dataset_keys=["actions"],
                load_next_obs=False,
                frame_stack=1,
                seq_length=seq_len,                  # length-10 temporal sequences
                pad_frame_stack=True,
                pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
                get_pad_mask=False,
                goal_mode=None,
                hdf5_cache_mode=hdf5_cache_mode,          # cache dataset in memory to avoid repeated file i/o
                hdf5_use_swmr=False,
                hdf5_normalize_obs=None,
                filter_by_attribute=filter_key,       # can optionally provide a filter key here
            )
    return dataset