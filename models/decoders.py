from torch import nn
from models.modules import *


class BCDecoder(nn.Module):
    def __init__(self,
                 shape_meta,
                 input_shape,
                 policy_output_head,
                 group):
        super().__init__()
        input_dim = input_shape[-1]

        self.group = eval(group.network)(**group.network_kwargs)
        policy_input_dim  = self.group.output_shape(input_shape, shape_meta)
        policy_output_head.network_kwargs.output_dim = shape_meta["ac_dim"]
        self.policy_output_head = eval(policy_output_head.network)(input_dim=policy_input_dim[0],
                                                                   **policy_output_head.network_kwargs)
        
    def forward(self, x, obs_dict):
        out = self.group(x, obs_dict)
        out = self.policy_output_head(out)
        return out


class ActionTokenOutput(nn.Module):
    """
    Outputing actions based on the latent output vector of the action token
    """
    def __init__(self,
                 shape_meta,
                 input_shape,
                 transformer_encoder,
                 policy_output_head,
                 group):
        super().__init__()
        self.group = eval(group.network)(**group.network_kwargs)
        input_shape  = self.group.output_shape(input_shape, shape_meta)
        input_dim = input_shape[-1]
        # seq_length = input_shape[0]
        self.transformer_encoder = eval(transformer_encoder.network)(input_dim=input_dim,
                                                                     **transformer_encoder.network_kwargs)
        self.policy_output_head = eval(policy_output_head.network)(input_dim=input_dim,
                                                                   **policy_output_head.network_kwargs)
        
    def forward(self, x, obs_dict):
        out = self.group(x, obs_dict)
        out = self.transformer_encoder(out)
        out = out[:, 0, ...]
        out = self.policy_output_head(out)
        return out