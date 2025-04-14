import torch
from torch import nn
import torch.nn.functional as F

from compas.pos_emb import get_sin_pos_enc
from compas.transition_action_adapter import ContinuousActionAdapter


class CompasParLayer(nn.TransformerEncoderLayer):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Module.
    This module models the temporal dynamics and object interactions in a dissentangled manner by
    applying object- and time-attention in parallel.

    Args:
    -----
    d_model: int
        Dimensionality of the input tokens
    nhead: int
        Number of heads in multi-head attention
    dim_feedforward: int
        Hidden dimension in the MLP
    dropout: float
        Amount of dropout to apply. Default is 0.1
    activation: int
        Nonlinear activation in the MLP. Default is ReLU
    layer_norm_eps: int
        Epsilon value in the layer normalization components
    batch_first: int
        If True, shape is (B, num_tokens, token_dim); otherwise, it is (num_tokens, B, token_dim)
    norm_first: int
        If True, transformer is in mode pre-norm: otherwise, it is post-norm
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=True, device=None, dtype=None):
        """
        Module initializer
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )

        self.self_attn_obj = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs
        )
        self.self_attn_time = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs
        )
        self.act_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs
        )

        self.act_norm = nn.LayerNorm(
            d_model,
            eps=layer_norm_eps,
        )

    def forward(self, src, action, time_mask=None):

        """
        Forward pass through the Object-Centric Transformer-v2.
        Overloads PyTorch's transformer forward pass.

        Args:
        -----
        src: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), self.act_norm(action), time_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, action, time_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def gen_act_causal_mask(self, slots: torch.Tensor) -> torch.Tensor:
        time_steps = slots.size(1)
        num_slots = slots.size(2)
        slots_mask = nn.Transformer.generate_square_subsequent_mask(time_steps,
                                                                    device=slots.device,)
        slots_mask = slots_mask.repeat_interleave(num_slots, 0)
        # slots_mask = slots_mask.repeat_interleave(num_slots, 1)
        return slots_mask

    def _sa_block(self, x, action, time_mask):
        """
        Forward pass through the parallel attention branches
        """
        B, num_imgs, num_slots, dim = x.shape

        # object-attention
        x_aux = x.clone().view(B * num_imgs, num_slots, dim)
        x_obj = self.self_attn_obj(
            query=x_aux,
            key=x_aux,
            value=x_aux,
            need_weights=False
        )[0]
        x_obj = x_obj.view(B, num_imgs, num_slots, dim)

        # action-attention
        attn_mask = self.gen_act_causal_mask(x)
        x_aux = x.clone().view(B, num_imgs * num_slots, dim)
        x_act = self.act_attn(
            query=x_aux,
            key=action,
            value=action,
            attn_mask=attn_mask,
            need_weights=False
        )[0]
        x_act = x_act.view(B, num_imgs, num_slots, dim)

        # time-attention
        x = x.transpose(1, 2).reshape(B * num_slots, num_imgs, dim)
        x_time = self.self_attn_time(
            query=x,
            key=x,
            value=x,
            attn_mask=time_mask,
            need_weights=False
        )[0]
        x_time = x_time.view(B, num_slots, num_imgs, dim).transpose(1, 2)

        y = self.dropout1(x_obj + x_time + x_act)
        return y


class CompasSeqLayer(nn.Module):
    """
    Sequential Object-Centric Video Prediction (OCVP-Seq) Transformer Layer.
    Sequentially applies object- and time-attention.

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    hidden_dim: int
        Hidden dimensionality of the MLPs in the transformer modules
    n_heads: int
        Number of heads for multi-head self-attention.
    """

    def __init__(self, d_model=128, dim_feedforward=256, nhead=4, batch_first=True,
                 norm_first=True, ):
        """
        Module initializer
        """
        super().__init__()
        self.token_dim = d_model
        self.hidden_dim = dim_feedforward
        self.nhead = nhead

        self.object_encoder_block = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.nhead,
            batch_first=True,
            norm_first=True,
            dim_feedforward=dim_feedforward
        )
        self.time_encoder_block = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.nhead,
            batch_first=True,
            norm_first=True,
            dim_feedforward=dim_feedforward
        )

        self.action_attention_block = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=self.nhead,
            batch_first=True,
            norm_first=True,
            dim_feedforward=dim_feedforward
        )

        return

    def forward(self, inputs, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-V1 Layer

        Args:
        -----
        inputs: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        B, num_imgs, num_slots, dim = inputs.shape

        # object-attention block. Operates on (B * N_imgs, N_slots, Dim)
        inputs = inputs.reshape(B * num_imgs, num_slots, dim)
        object_encoded_out = self.object_encoder_block(inputs)
        object_encoded_out = object_encoded_out.reshape(B, num_imgs, num_slots, dim)

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        object_encoded_out = object_encoded_out.reshape(B * num_slots, num_imgs, dim)
        object_encoded_out = self.time_encoder_block(object_encoded_out)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out


class CompasDynamicsModel(nn.Module):
    def __init__(self,
                 num_slots: int,
                 slots_dim: int,
                 tokens_dim: int,
                 max_timestep: int,
                 actions_dim: int,
                 projected_actions_dim: int,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 hidden_mult: int = 4,
                 parallel: bool = False):
        super().__init__()
        self.num_slots = num_slots
        self.slots_dim = slots_dim
        self.max_timestep = max_timestep
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_mult = hidden_mult
        self.act_adapter = ContinuousActionAdapter(actions_dim, projected_actions_dim)

        # self._fw_blocks = self._par_forward_blocks if parallel else self._seq_forward_blocks
        # self.next_slot_token = nn.Parameter(torch.randn(1, 1, 1, slots_dim) * slots_dim ** -0.5)

        # mlp_in_dim = slots_dim * 2
        # mlp_hidden_dim = slots_dim * 2 * hidden_mult

        # self.action_attacher = nn.Sequential(
        #     nn.Linear(mlp_in_dim, mlp_hidden_dim),
        #     nn.SELU(),
        #     nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        #     nn.SELU(),
        #     nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        #     nn.SELU(),
        #     nn.Linear(mlp_hidden_dim, slots_dim),
        # )

        # self.slots_action_blocks = nn.ModuleList(
        #     [ActionAttentionBlock(slots_dim, num_heads, hidden_mult) for _ in range(num_layers)]
        # )
        #
        # self.slots_pred_blocks = nn.ModuleList(
        #     [SlotsDynamicsBlock(slots_dim, num_heads, hidden_mult) for _ in range(num_layers)]
        # )
        self.in_proj = nn.Linear(slots_dim, tokens_dim, bias=False)

        self.blocks = nn.ModuleList([
            CompasParLayer(
                d_model=tokens_dim,
                nhead=num_heads,
                batch_first=True,
                norm_first=True,
                dim_feedforward=tokens_dim * hidden_mult, )
        ])

        self.time_pos_encoding = nn.Parameter(
            get_sin_pos_enc(max_timestep, tokens_dim), requires_grad=False)

        # mlp_hidden_dim = slots_dim * hidden_mult

        self.mlp_out = nn.Sequential(
            nn.Linear(tokens_dim, slots_dim),
            # nn.SELU(),
            # nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            # nn.SELU(),
            # nn.Linear(mlp_hidden_dim, slots_dim),
        )

    def load_state_from_method(self, state_dict: dict):
        state_dict = {k.replace('transition_model.', ''): v for k, v in state_dict.items() if
                      k.startswith('transition_model.')}
        self.load_state_dict(state_dict)

    def auto_predict_trajectory(self, slots: torch.Tensor, actions: torch.Tensor):
        pred_len = slots.size(1) - self.max_timestep
        return self.predict_trajectory(slots, actions, length=pred_len), self.max_timestep

    # def _attach_next_slots(self, slots: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    #     batch_size = slots.size(0)
    #     slots_token = self.next_slot_token.repeat(batch_size, 1, self.num_slots, 1)
    #     slots_token = slots_token + self.slots_pos_encoding.repeat(batch_size, 1, 1, 1)
    #     # slots_token = self.action_attacher(torch.cat((slots_token, action), dim=-1))
    #     return torch.cat((slots, slots_token), dim=1)
    #
    #
    # def _attach_next_slots(self, slots: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    #     batch_size = slots.size(0)
    #     slots_token = self.next_slot_token.repeat(batch_size, 1, self.num_slots, 1)
    #     slots_token = slots_token + self.slots_pos_encoding.repeat(batch_size, 1, 1, 1)
    #     # slots_token = self.action_attacher(torch.cat((slots_token, action), dim=-1))
    #     return torch.cat((slots, slots_token), dim=1)

    def predict_trajectory(self, slots: torch.Tensor, actions: torch.Tensor, length: int = 2):
        next_slots = []
        # print(slots.shape)
        max_t = min(self.max_timestep, slots.size(1))
        slots = slots[:, :max_t]
        offset = 0

        # print(actions.shape)
        for i in range(length):
            act_step = i - offset
            act = actions[:, act_step:max_t + act_step]
            next_slot = self.forward(slots, act)
            if max_t == self.max_timestep:
                slots = slots[:, 1:]
            else:
                max_t += 1
                offset += 1
            slots = torch.cat((slots, next_slot.unsqueeze(1)), dim=1)
            next_slots.append(next_slot)

        return torch.stack(next_slots, dim=1)

    def _par_forward_blocks(self, slots: torch.Tensor, actions: torch.Tensor):
        blocks = zip(self.slots_action_blocks, self.slots_pred_blocks)
        for act_block, pred_block in blocks:
            act_slots = act_block(slots, actions)
            pred_slots = pred_block(slots)
            slots = act_slots + pred_slots
        return slots

    def _seq_forward_blocks(self, slots: torch.Tensor, actions: torch.Tensor):
        blocks = zip(self.slots_action_blocks, self.slots_pred_blocks)
        for act_block, pred_block in blocks:
            slots = act_block(slots, actions)
            slots = pred_block(slots)
        return slots

    def forward(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = slots.shape
        # slots = self._attach_next_slots(slots, B, actions)
        in_slots = slots
        slots = self.in_proj(slots)

        time_enc = self.time_pos_encoding[:, -T:]
        actions = self.act_adapter(actions)

        slots = slots + time_enc.unsqueeze(2).expand(B, -1, self.num_slots, -1)
        actions = actions + time_enc.expand(B, -1, -1)

        for block in self.blocks:
            slots = block(slots, actions)

        slots = in_slots + self.mlp_out(slots)
        return slots[:, -1]
