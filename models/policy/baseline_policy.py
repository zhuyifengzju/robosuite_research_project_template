import torch
from models.modules import *
from models.decoders import *
from models.policy.base_policy import *

class BCGMMPolicy(BasePolicy):
    """It has the same architecture, with the only difference of needing to sample actions."""
    def __init__(self,
                 policy_cfg,
                 shape_meta):
        super().__init__(policy_cfg, shape_meta)
        self.data_aug = eval(policy_cfg.data_aug.network)(**policy_cfg.data_aug.network_kwargs)
        self.policy_cfg = policy_cfg
        input_shape = shape_meta["all_shapes"]["stacked_rgb"]
        policy_cfg.img_aug.network_kwargs["input_shape"] = input_shape
        if "img_c" in policy_cfg.img_aug.network_kwargs:
            policy_cfg.img_aug.network_kwargs["img_c"] = input_shape[0]
        self.img_aug = eval(policy_cfg.img_aug.network)(**policy_cfg.img_aug.network_kwargs)
        input_shape = self.img_aug.output_shape(input_shape)

        policy_cfg.encoder.network_kwargs["input_shape"] = input_shape        
        self.encoder = eval(policy_cfg.encoder.network)(**policy_cfg.encoder.network_kwargs)
        print(input_shape)
        input_shape = self.encoder.output_shape(input_shape)
        # You need compute spatial scale for roi align
        print(input_shape)

        policy_cfg.decoder.network_kwargs.input_shape = input_shape
        policy_cfg.decoder.network_kwargs["group"] = policy_cfg.group
        policy_cfg.decoder.network_kwargs["policy_output_head"] = policy_cfg.policy_output_head
        self.decoder = eval(policy_cfg.decoder.network)(shape_meta, **policy_cfg.decoder.network_kwargs)

    def forward_fn(self, data):
        out = self.img_aug(data["obs"]["stacked_rgb"])

        out = self.data_aug(out)
        
        self.encoder_out = self.encoder(out)

        self.decoder_out = self.decoder(self.encoder_out, data["obs"])

        return self.decoder_out

    def forward(self, data):
        data = self.process_input_for_training(data)        
        decoder_out = self.forward_fn(data)
        return decoder_out

    def get_action(self, data):
        data = TensorUtils.to_device(data, self.device)
        data = self.process_input_for_evaluation(data)
        with torch.no_grad():
            action = self.forward_fn(data)
        return action.detach().cpu().squeeze().numpy()


    def process_input_for_training(self, x):
        return TensorUtils.recursive_dict_list_tuple_apply(
            x,
            {
                torch.Tensor: lambda x: x.squeeze(dim=1),
            }
        )

    def process_input_for_evaluation(self, x):
        return TensorUtils.recursive_dict_list_tuple_apply(
            x,
            {
                torch.Tensor: lambda x: x.unsqueeze(dim=0),
            }
        )

    @property
    def device(self):
        return next(self.parameters()).device    


class BCTransformerPolicyRGB(BasePolicy):
    def __init__(self,
                 policy_cfg,
                 shape_meta):
        super().__init__(policy_cfg,
                         shape_meta)
        
        self.policy_cfg = policy_cfg
        input_shape = shape_meta["all_shapes"]["agentview_rgb"]
        obs_keys = list(shape_meta["all_shapes"].keys())
        color_aug = eval(policy_cfg.color_aug.network)(**policy_cfg.color_aug.network_kwargs)        
        self.img_names = []
        self.input_img_shapes = []
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                self.img_names.append(name)
        for img_name in self.img_names:
            self.input_img_shapes.append(shape_meta["all_shapes"][img_name])
        policy_cfg.translation_aug.network_kwargs["input_shape"] = self.input_img_shapes[0]
        translation_aug = eval(policy_cfg.translation_aug.network)(**policy_cfg.translation_aug.network_kwargs)        
        self.img_aug = DataAugGroup([color_aug, translation_aug])
        
        self.rgb_encoder_dict = {}
        self.spatial_projection_dict = {}
        self.projection_output_shapes = {}
        for img_name in self.img_names:
            self.rgb_encoder_dict[img_name] = eval(policy_cfg.rgb_encoder.network)(**policy_cfg.rgb_encoder.network_kwargs)
            rgb_output_shape = self.rgb_encoder_dict[img_name].output_shape(shape_meta["all_shapes"][img_name])
            policy_cfg.spatial_projection.network_kwargs["input_shape"] = rgb_output_shape
            self.spatial_projection_dict[img_name] = eval(policy_cfg.spatial_projection.network)(**policy_cfg.spatial_projection.network_kwargs)
            self.projection_output_shapes[img_name] = self.spatial_projection_dict[img_name].output_shape(rgb_output_shape)

        self.rgb_encoder_dict = nn.ModuleDict(self.rgb_encoder_dict)
        self.spatial_projection_dict = nn.ModuleDict(self.spatial_projection_dict)
        self.group = eval(policy_cfg.group.network)(**policy_cfg.group.network_kwargs)
        input_shape = self.group.output_shape(list(self.projection_output_shapes.values()))

        policy_cfg.temporal_position.network_kwargs.input_shape = input_shape
        self.temporal_position_encoding = eval(policy_cfg.temporal_position.network)(**policy_cfg.temporal_position.network_kwargs)
        input_shape = self.temporal_position_encoding.output_shape(input_shape)

        # Initialize transformer
        input_dim = input_shape[-1]
        self.transformer = eval(policy_cfg.transformer.network)(input_dim=input_dim,
                                                                **policy_cfg.transformer.network_kwargs)
        # If we use MLP only, we will concatenate tensor in the policy model
        policy_cfg.policy_output_head.network_kwargs.input_dim = input_shape[-1]
        policy_cfg.policy_output_head.network_kwargs.output_dim = shape_meta["ac_dim"]
        self.policy_output_head = eval(policy_cfg.policy_output_head.network)(**policy_cfg.policy_output_head.network_kwargs)

    def get_img_tuple(self, data):
        img_tuple = tuple([data["obs"][img_name] for img_name in self.img_names])
        return img_tuple
    
    def get_aug_output_dict(self, out):
        img_dict = {img_name: out[idx] for idx, img_name in enumerate(self.img_names)}
        return img_dict

    def encode_fn(self, data):
        assert(self.reset_at_start), "The policy needs to be reset at least once!!!"
        batch_size = data["obs"][self.img_names[0]].shape[0]
        out = self.get_aug_output_dict(self.img_aug(self.get_img_tuple(data)))
        
        latent_outputs = []
        for img_name in self.img_names:
            latent_outputs.append(self.spatial_projection_dict[img_name](self.rgb_encoder_dict[img_name](out[img_name])))
        self.position_embedding_out = self.group(latent_outputs, data["obs"])
        return self.position_embedding_out        

    def decode_fn(self, x, per_step=False):
        # print(x.shape, self.temporal_position_encoding)
        self.temporal_positions = self.temporal_position_encoding(x)
        self.temporal_out = x + self.temporal_positions.unsqueeze(0).unsqueeze(2)
        original_shape = self.temporal_out.shape
        self.transformer.compute_mask(self.temporal_out.shape)
        flattened_temporal_out = TensorUtils.join_dimensions(self.temporal_out, 1, 2)
        transformer_out = self.transformer(flattened_temporal_out)
        transformer_out = transformer_out.reshape(original_shape)
        action_token_out = transformer_out[:, :, 0, :]
        if per_step:
            action_token_out = action_token_out[:, -1:, :]
        action_outputs = self.policy_output_head(action_token_out)
        return action_outputs

    def forward(self, data):
        data = self.process_input_for_training(data)
        out = TensorUtils.time_distributed(data, self.encode_fn)
        batch_size, seq_len = out.shape[:2]
        dist = self.decode_fn(out)
        return dist
    
    def get_action(self, data):
        data = TensorUtils.to_device(data, self.device)
        data = self.process_input_for_evaluation(data)
        with torch.no_grad():
            # encode_out = self.encode_fn(data)
            encode_out = TensorUtils.time_distributed(data, self.encode_fn)
            self.queue.append(encode_out)
            if len(self.queue) > self.max_len:
                self.queue.pop(0)
            temporal_sequence = torch.cat(self.queue, dim=1)
            dist = self.decode_fn(temporal_sequence, per_step=True)
        return dist.sample().detach().cpu().squeeze().numpy()
        
    def reset(self):
        self.queue = []
        self.reset_at_start = True

    def process_input_for_training(self, x):
        return x

    def process_input_for_evaluation(self, x):
        return TensorUtils.recursive_dict_list_tuple_apply(
            x,
            {
                torch.Tensor: lambda x: x.unsqueeze(dim=0).unsqueeze(dim=0),
            }
        )

    @property
    def device(self):
        return next(self.parameters()).device