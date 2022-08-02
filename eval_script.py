import os
import argparse
from utils import s3_utils, log_utils
from envs import *
import utils.utils as utils
from models.modules import safe_cuda
from models.policy import *
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from easydict import EasyDict
import h5py
import json
import imageio
from tqdm import trange
from pathlib import Path
utils.initialize_env()

class ObservationPreProcessor():
    """This is a class to convert from robosuite observation to data input to the neural network model."""
    def __init__(self, cfg):
        self.cfg = cfg


        self.rgb_mapping = {"agentview_rgb": "agentview_image",
                               "eye_in_hand_rgb": "robot0_eye_in_hand_image"}
        self.depth_mapping = {"agentview_depth": "agentview_depth",
                              "eye_in_hand_depth": "robot0_eye_in_hand_depth"}

        self.proprio_key_mapping = {"gripper_states": "robot0_gripper_qpos",
                                    "joint_states": "robot0_joint_pos",
                                    "ee_pos": "robot0_eef_pos",
                                    "ee_ori": "robot0_eef_quat"}

        log_utils.warning_print("Remember to reset this")

    def reset(self):
        self.gripper_history = []

    def get_obs(self, obs, env=None, use_rgb=True, use_depth=False):
        """
        args:
           obs (dict): observation dictionary from robosuite
           use_rgb (bool): Process rgb images. 
           use_depth (bool): Process depth images. 
        """
        data = {"obs": {}}

        if use_depth:
            assert(env is not None)
        if use_rgb:
            data["obs"]["agentview_rgb"] = ObsUtils.process_obs(torch.from_numpy(obs[self.rgb_mapping["agentview_rgb"]]), obs_key="agentview_rgb")
            data["obs"]["eye_in_hand_rgb"] = ObsUtils.process_obs(torch.from_numpy(obs[self.rgb_mapping["eye_in_hand_rgb"]]), obs_key="eye_in_hand_rgb")
            # data["obs"]["agentview_rgb"] = utils.process_image_input(torch.from_numpy(np.array(obs[self.rgb_mapping["agentview_rgb"]]).transpose(2, 0, 1)).float())
            # data["obs"]["eye_in_hand_rgb"] = utils.process_image_input(torch.from_numpy(np.array(obs[self.rgb_mapping["eye_in_hand_rgb"]]).transpose(2, 0, 1)).float())

        if use_depth:
            data["obs"]["agentview_depth"] = ObsUtils.process_obs(torch.from_numpy(utils.get_normalized_depth(env.sim, obs[self.depth_mapping["agentview_depth"]])), obs_key="agentview_depth")
            data["obs"]["eye_in_hand_depth"] = ObsUtils.process_obs(torch.from_numpy(utils.get_normalized_depth(env.sim, obs[self.depth_mapping["eye_in_hand_depth"]])), obs_key="eye_in_hand_depth")

        if self.gripper_history == []:
            for _ in range(5):
                self.gripper_history.append(torch.from_numpy(obs["robot0_gripper_qpos"]))
                self.gripper_history.pop(0)
                self.gripper_history.append(torch.from_numpy(obs["robot0_gripper_qpos"]))
                data["obs"]["gripper_history"] = torch.cat(self.gripper_history, dim=-1).float()

        for proprio_state_key, obs_key in self.proprio_key_mapping.items():
            data["obs"][proprio_state_key]= torch.from_numpy(obs[obs_key]).float()
        return data
        

        
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint-dir',
        type=str
    )
    parser.add_argument(
        '--eval-horizon',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--num-eval',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--task-name',
        type=str,
        default="normal",
    )    
    parser.add_argument(
        '--remote-folder-name',
        type=str,
        default="yifengz/gb_bc"
    )
    parser.add_argument(
        '--bucket-name',
        type=str,
        default="tri-gb"
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=7
    )
    parser.add_argument(
        '--record-video',
        action='store_true'
    )
    parser.add_argument(
        '--use-terminal',
        action='store_true'
    )
    return parser.parse_args()

def download_checkpoint_dir(args):
    s3_interface = s3_utils.S3Interface()
    s3_interface.download_folder(args.checkpoint_dir,
                                 args.remote_folder_name,
                                 args.bucket_name)

def upload_folder(folder_name, args):
    s3_interface = s3_utils.S3Interface()
    s3_interface.upload_folder(folder_name,
                               args.remote_folder_name,
                               args.bucket_name)

def upload_file(file_name, args):
    s3_interface = s3_utils.S3Interface()
    s3_interface.upload_file(file_name,
                             args.remote_folder_name,
                             args.bucket_name)
    
    
def load_eval_model(cfg, shape_meta):
    model = safe_cuda(eval(cfg.algo.model.name)(cfg.algo.model, shape_meta))
    model_checkpoint = cfg.model_dir.model_checkpoint_name
    model_state_dict, _ = utils.torch_load_model(cfg.model_dir.model_checkpoint_name)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def initialize_obs_modalities(cfg):

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.algo.obs.modality})    
    all_obs_keys = []
    for modality_name, modality_list in cfg.algo.obs.modality.items():
        all_obs_keys += modality_list

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=cfg.data.params.data_file_name,
        all_obs_keys=all_obs_keys,
        verbose=True
    )
    return all_obs_keys, shape_meta

def eval_loop(cfg, model, suffix, args):

    n_eval = args.num_eval
    random_seed = args.random_seed
    eval_horizon = args.eval_horizon

    with h5py.File(cfg.data.params.data_file_name) as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    domain_name = env_args["domain_name"]
    env_kwargs = env_args["env_kwargs"]
    task_name = args.task_name
    env = TASK_MAPPING[domain_name](
        exp_name=task_name,
        **env_kwargs,
    )

    if env_kwargs["controller_configs"]["type"] == "OSC_POSE":
        action_dim = 7
    elif env_kwargs["controller_configs"]["type"] == "OSC_POSITION":
        action_dim = 4
    else:
        raise ValueError

    eval_range = trange(n_eval)

    utils.set_manual_seeds(random_seed)

    obs_processor = ObservationPreProcessor(cfg)
    num_success = 0

    video_dir = os.path.join(cfg.model_dir.output_dir, f"videos_{suffix}")
    state_dir = os.path.join(cfg.model_dir.output_dir, f"record_states_{suffix}")
    
    for i in eval_range:
        eval_range.set_description(f"{domain_name} - {task_name}, Success rate: {num_success} / {i}")

        model.reset()
        obs_processor.reset()


        env.reset()
        model_xml = env.sim.model.get_xml()
        initial_mjstate = env.sim.get_state().flatten()

        xml = utils.postprocess_model_xml(model_xml, {})
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_mjstate)
        env.sim.forward()
        for _ in range(5):
            env.step([0.] * (action_dim - 1) + [-1.])

        done = False
        record_states = []
        record_imgs = []
        steps = 0
        obs = env._get_observations()
        while not done and steps < eval_horizon:
            record_states.append(env.sim.get_state().flatten())
            if args.record_video:
                record_imgs.append(obs["agentview_image"])
            steps += 1

            data = obs_processor.get_obs(obs, env)
            action = model.get_action(data)

            obs, reward, done, info = env.step(action)
            done = env._check_success()

            # import cv2
            # cv2.imshow("", obs["agentview_image"]); cv2.waitKey(10)

            if done:
                num_success += 1
                for _ in range(10):
                    record_states.append(env.sim.get_state().flatten())
                    if args.record_video:
                        record_imgs.append(obs["agentview_image"])
                    env.sim.forward()

        if args.record_video:
            os.makedirs(video_dir, exist_ok=True)
            video_writer = imageio.get_writer(os.path.join(video_dir, f"{task_name}_{i}_{done}.mp4"), fps=60)
            for img in record_imgs:
                video_writer.append_data(img)
            video_writer.close()

        os.makedirs(state_dir, exist_ok=True)
        with h5py.File(os.path.join(state_dir, f"{task_name}_{i}_{done}.hdf5"), "w") as state_file:
            state_file.attrs["domain_name"] = domain_name
            state_file.attrs["task_name"] = task_name

            state_file.create_dataset("states", data=np.array(record_states))


    if args.record_video:
        upload_folder(video_dir, args)
    upload_folder(state_dir, args)
    return num_success


def main():
    args = parse_args()

    checkpoint_dir = args.checkpoint_dir
    print(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        log_utils.warning_print("Downloading checkpoints")
        download_checkpoint_dir(args)
    assert(os.path.exists(os.path.join(checkpoint_dir, "finished.log"))), log_utils.error_print("This model checkpoint hasn't finished the full training process")

    # Load config
    with open(os.path.join(checkpoint_dir, "cfg.json"), "r") as f:
        cfg = EasyDict(json.load(f))

    logger_manager = log_utils.LoggerManager(cfg, train_mode=False)
    if not args.use_terminal:
        logger_manager.use_logger()
    

    _, shape_meta = initialize_obs_modalities(cfg)
    print(shape_meta)
    # Load model

    for model_name in Path(checkpoint_dir).glob("*pth"):
        suffix = str(model_name).split("/")[-1].replace(".pth", "").split("_")
        if len(suffix) == 1:
            suffix = "final"
        else:
            suffix = f"{int(suffix[-1]):03}"
        result_file = os.path.join(checkpoint_dir, f"result_{args.task_name}_{suffix}.json")
        if os.path.exists(result_file):
            print("The checkpoint has already been evluated. Skipping")
            continue
        print(f"{model_name}")        
        cfg.model_dir.model_checkpoint_name = model_name
        model = load_eval_model(cfg, shape_meta)
        num_success = eval_loop(cfg, model, suffix, args)
        with open(os.path.join(result_file), "w+") as f:
            json.dump(num_success, f)
        upload_file(result_file, args)
    # Upload results to both s3 bucket and wandb (optional)

if __name__ == "__main__":
    main()
