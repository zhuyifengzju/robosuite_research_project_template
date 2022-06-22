





def train(cfg):
    data_path = cfg.data.params.data_file_name

    # ObsUtils.ImageModality.set_obs_processor(processor=stack_obs_processor)
    # ObsUtils.ImageModality.set_obs_unprocessor(unprocessor=stack_obs_unprocessor)    
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.algo.obs.modality})
    
    # ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.algo.obs.modality})

    all_obs_keys = []
    for modality_name, modality_list in cfg.algo.obs.modality.items():
        all_obs_keys += modality_list    