import os
import hydra
from omegaconf import OmegaConf
import pprint
import yaml
from easydict import EasyDict
from utils import path_utils, log_utils, train_utils, s3_utils
import utils.utils as utils
from copy import deepcopy
import wandb
from torch.utils.data import Dataset, DataLoader
from models.modules import safe_cuda
from models.policy import *
from models.loss_fn import *
from tqdm import trange
from einops import rearrange
import json


torch.cuda.empty_cache()
torch.cuda.synchronize()
utils.initialize_env()

def train(cfg):

    # Initialize dataset
    dataset, shape_meta = train_utils.get_dataset(dataset_path=cfg.data.params.data_file_name, 
                                                  obs_modality=cfg.algo.obs.modality, 
                                                  seq_len=cfg.algo.train.rnn_horizon if cfg.algo.train.use_rnn else 1, 
                                                  filter_key=cfg.data.params.filter_key, 
                                                  hdf5_cache_mode=cfg.hdf5_cache_mode)
    dataloader = DataLoader(dataset, batch_size=cfg.algo.train.batch_size, shuffle=True, num_workers=cfg.algo.train.num_workers)

    # Initialize the model
    model = safe_cuda(eval(cfg.algo.model.name)(cfg.algo.model,
                                                shape_meta))
    model.reset()

    print(model)

    # Initialize the loss function
    loss_fn = eval(cfg.algo.loss_fn.fn)(**cfg.algo.loss_fn.loss_kwargs)

    # Initialize the optimizer
    optimizer = eval(cfg.algo.optimizer.name)(model.parameters(), lr=cfg.algo.train.lr, **cfg.algo.optimizer.parameters)

    # Initialize the scheduler
    if cfg.algo.scheduler is not None:
        if cfg.algo.scheduler.name is not None:
            scheduler = eval(cfg.algo.scheduler.name)(optimizer,
                                                      **cfg.algo.scheduler.parameters)
        else:
            scheduler = None

    train_range = trange(cfg.algo.train.n_epochs)

    best_loss = None
    for epoch in train_range:
        model.train()
        training_loss = []
        num_iters = len(dataloader)
        for (idx, data) in enumerate(dataloader):
            data = TensorUtils.to_device(data, model.device)
            output = model(data)

            # if idx == 1:
            #     img_grid = model.aug_out["agentview_rgb"]
            #     img_grid = rearrange(img_grid, "b t c h w -> (b t) c h w", t=10)
            #     img_grid = torchvision.utils.make_grid(img_grid, nrow=40)
            #     images = wandb.Image(img_grid)
            #     wandb.log({"Augmented images": images})

            if cfg.algo.train.use_rnn:
                loss = loss_fn(output, data["actions"])
            else:
                loss = loss_fn(output, data["actions"].squeeze(1))

            optimizer.zero_grad()
            loss.backward()

            if cfg.algo.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.algo.train.grad_clip)
            optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch + idx / num_iters)

            training_loss.append(loss.item())

        training_loss = np.mean(training_loss)
        wandb.log({"training loss": training_loss, "epoch": epoch})

        if epoch % 5 == 0:
            model.eval()

            overall_training_loss = []
            for (idx, data) in enumerate(dataloader):
                data = TensorUtils.to_device(data, model.device)
                with torch.no_grad():
                    output = model(data)

                if cfg.algo.train.use_rnn:
                    loss = loss_fn(output, data["actions"])
                else:
                    loss = loss_fn(output, data["actions"].squeeze(1))

                overall_training_loss.append(loss.item())

            overall_training_loss = np.mean(overall_training_loss)
            wandb.log({"overall training loss": overall_training_loss, "epoch": epoch})

            if best_loss is None or best_loss > overall_training_loss:
                best_loss = overall_training_loss
                model_checkpoint_name = cfg.model_dir.model_checkpoint_name.replace(".pth", f"_{epoch}.pth")
                utils.torch_save_model(model, model_checkpoint_name, cfg=cfg)

        train_range.set_description(f"Current loss: {np.round(overall_training_loss, 3)}, Best loss: {np.round(best_loss, 3)}")

    with open(f"{cfg.model_dir.output_dir}/finished.log", "w") as f:
        pass
    
    return model
    


@hydra.main(config_path="./configs", config_name="config")
def main(hydra_cfg):
    # Initialize configs
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    s3_interface = s3_utils.S3Interface()    
    # Download datasets if the dataset is not detected
    pp = pprint.PrettyPrinter(indent=2)

    # Initialize checkpoint path
    # with open("result_path_list.json", "r") as f:
    #     result_path_list = json.load(f)

    result_path_list = s3_interface.list("results")
    path_utils.checkpoint_model_dir(cfg, additional_dir_list=result_path_list)
    # Make a placeholder on s3 to avoid name conflict
    s3_interface.upload_folder(cfg.model_dir.output_dir)

    dataset_path = cfg.data.params.data_file_name
    if not os.path.exists(dataset_path):
        print("Downloading data")
        s3_interface.download_file(dataset_path)
    
    logger_manager = log_utils.LoggerManager(cfg)
    pp.pprint(cfg)
    
    if cfg.flags.use_logger:
        logger_manager.use_logger()
    print(pp.pformat(cfg))
    # Start training

    wandb_mode = "online"        
    if cfg.flags.debug:
        wandb_mode = "disabled"
    wandb.init(project=cfg.wandb_project, config=cfg, mode=wandb_mode)
    wandb.run.name = cfg.model_dir.output_dir.replace("./results", "")

    train(cfg)
    # Conclude

    # Upload model to s3
    s3_interface.upload_folder(cfg.model_dir.output_dir)
    
    wandb.finish()

if __name__ == "__main__":
    main()
    
