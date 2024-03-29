import os
import xml.etree.ElementTree as ET
import robosuite
from robosuite.utils.mjcf_utils import find_elements
import numpy as np
import json
import torch
import random
from models.modules import safe_cuda
from pathlib import Path
from robosuite.utils import camera_utils

def initialize_env():
    os.environ["WANDB_API_KEY"] = "583f8684883901e7b048ee514f3551b4deeb2d76"
    import robosuite.utils.macros as macros
    macros.IMAGE_CONVENTION = "opencv"

def postprocess_model_xml(xml_str, cameras_dict={}):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.

    Args:
        xml_str (str): Mujoco sim demonstration XML file as string

    Returns:
        str: Post-processed xml file as string
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        if "robosuite" not in old_path_split:
            continue
        ind = max(
            loc for loc, val in enumerate(old_path_split) if val == "robosuite"
        )  # last occurrence index
        new_path_split = path_split + old_path_split[ind + 1 :]
        new_path = "/".join(new_path_split)
        elem.set("file", new_path)

    # cameras = root.find("worldbody").findall("camera")
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        camera_name = camera.get("name")
        if camera_name in cameras_dict:
            camera.set("name", camera_name)
            camera.set("pos", cameras_dict[camera_name]["pos"])
            camera.set("quat", cameras_dict[camera_name]["quat"])
            camera.set("mode", "fixed")

        
    return ET.tostring(root, encoding="utf8").decode("utf8")



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def torch_save_model(policy, model_path, cfg=None):
    torch.save({"state_dict": policy.state_dict(), "cfg": cfg}, model_path)
    with open(model_path.replace(".pth", ".json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

def torch_load_model(model_path):
    model_dict = torch.load(model_path)
    cfg = None
    if "cfg" in model_dict:
        cfg = model_dict["cfg"]
    return model_dict["state_dict"], cfg

def save_run_cfg(output_dir, cfg):
    with open(os.path.join(output_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

def load_run_cfg(output_dir, config_name="cfg"):
    with open(os.path.join(output_dir, f"{config_name}.json"), "r") as f:
        return json.load(f)

def set_manual_seeds(seed: int, deterministic: bool = False):
    if seed is not None:
        assert(type(seed) is int)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)            
        print(f"Setting random seeds: {seed}")
    else:
        print(f"Not setting random seeds. The experiment might not be reproducible.")

def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.

def reconstruct_image_output(img_array):
    # return (img_array + 1.) / 2. * 255.
    return img_array * 255.
   
def update_env_kwargs(env_kwargs, **kwargs):
    for (k, v) in kwargs.items():
        env_kwargs[k] = v


def create_run_model(cfg, output_dir, additional_dir_list=None):
    experiment_id = 0
    for path in Path(output_dir).glob('run_*'):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split('run_')[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    if additional_dir_list is not None:
        for path in additional_dir_list:
            if output_dir.replace("./", "") != str(Path(path).parents[0]).replace("./", ""):
                # print("They are different", output_dir.replace("./", ""), " | ", str(Path(path)).replace("./", ""))
                continue
            try:
                folder_id = int(str(path).split('run_')[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass

    experiment_id += 1
    output_dir += f"/run_{experiment_id:03d}"
    os.makedirs(output_dir, exist_ok=True)
    # utils.save_run_cfg(output_dir, cfg)

    return output_dir

def get_normalized_depth(env_sim, depth_img):
    """Normalize the depth image to range from [0., 1.]"""
    real_depth_img = camera_utils.get_real_depth_map(env_sim, depth_img)
    min_depth = real_depth_img.min()
    max_depth = real_depth_img.max()
    return (real_depth_img - min_depth) / (max_depth - min_depth)
