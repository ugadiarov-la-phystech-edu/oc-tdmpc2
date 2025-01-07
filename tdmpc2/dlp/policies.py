from types import SimpleNamespace
from typing import Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Modules
"""


class ParticleAttention(nn.Module):
    """
    particle-based multi-head masked attention layer with output projection
    """

    def __init__(self, embed_dim, n_head, attn_pdrop=0.0, resid_pdrop=0.0, att_type='hybrid', linear_bias=False):
        super().__init__()
        assert embed_dim % n_head == 0
        assert att_type in ['hybrid', 'cross', 'self']
        self.att_type = att_type
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.n_head = n_head

    def forward(self, x, c=None, mask=None, return_attention=False):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)

        query_input = x
        if self.att_type == 'hybrid':
            key_value_input = torch.cat([x, c], dim=1)
            key_value_N = key_value_input.shape[1]
        elif self.att_type == 'cross':
            key_value_input = c
            key_value_N = key_value_input.shape[1]
        else:   # self.att_type == 'self'
            key_value_input = x
            key_value_N = N

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key_value_input).view(B, key_value_N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, key_value_N * T, hs)
        q = self.query(query_input).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(key_value_input).view(B, key_value_N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, key_value_N * T, hs)
        # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, key_value_N * T)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)
            att.masked_fill_(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        if return_attention:
            attention_matrix = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, N*T, key_value_N*T) x (B, nh, key_value_N*T, hs) -> (B, nh, N*T, hs)
        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)

        # return
        if return_attention:
            return y, attention_matrix
        else:
            return y


class EITBlock(nn.Module):
    def __init__(self, embed_dim, h_dim, n_head, attn_pdrop=0.1, resid_pdrop=0.1, att_type='self'):
        super().__init__()
        self.att_type = att_type

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        if self.att_type != 'self':
            self.ln_c = nn.LayerNorm(embed_dim)

        self.attn = ParticleAttention(embed_dim, n_head, attn_pdrop, resid_pdrop, att_type)

        self.mlp = nn.Sequential(nn.Linear(embed_dim, h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim, embed_dim),
                                 nn.Dropout(resid_pdrop))

    def forward(self, x_in, c=None, x_mask=None, c_mask=None, return_attention=False):

        mask = x_mask

        if self.att_type != 'self':
            c = self.ln_c(c)

        if return_attention:
            x, attention_matrix = self.attn(self.ln1(x_in), c, mask, return_attention)
            x = x + x_in
        else:
            x = x_in + self.attn(self.ln1(x_in), c, mask)

        x = x + self.mlp(self.ln2(x))

        if return_attention:
            return x, attention_matrix
        else:
            return x


"""
Entity Interaction Transformer Policy
"""


class EITActor(nn.Module):
    def __init__(self, cfg, particle_fdim, action_dim, background_fdim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masking = cfg.eit_masking
        embed_dim = cfg.eit_embed_dim
        h_dim = cfg.eit_h_dim
        n_head = cfg.eit_n_head
        dropout = cfg.eit_dropout
        particle_dim = particle_fdim - 1 if self.masking else particle_fdim
        self.particle_projection = nn.Linear(particle_dim, embed_dim)
        self.particle_self_att1 = EITBlock(embed_dim, h_dim, n_head,
                                           attn_pdrop=dropout, resid_pdrop=dropout, att_type='self')
        self.particle_self_att2 = EITBlock(embed_dim, h_dim, n_head,
                                           attn_pdrop=dropout, resid_pdrop=dropout, att_type='self')
        self.particle_pool_att = EITBlock(embed_dim, h_dim, n_head,
                                          attn_pdrop=dropout, resid_pdrop=dropout, att_type='cross')

        self.background_dim = background_fdim
        self.use_background = cfg.eit_use_background
        if self.use_background:
            self.background_projection = nn.Linear(self.background_dim, embed_dim)

        self.ln = nn.LayerNorm(embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim, bias=True)

        self.output_mlp = nn.Sequential(nn.Linear(embed_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, 2 * action_dim))

        # special particle
        self.out_particle = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

    def forward(self, obs, return_attention=False, bg=None):
        particles = obs
        bs, n_particles, feature_dim = particles.shape
        if return_attention:
            attention_dict = {}

        # preprocess particles and produce masks
        state_mask, goal_mask = None, None
        if self.masking:
            # prepare attention masks (based on obj_on)
            particles_obj_on = particles[..., -1].view(bs, -1)

            if self.use_background:
                particles_obj_on = torch.cat([particles_obj_on, particles_obj_on.new_ones([bs, 1])], dim=-1)  # add special particles

            particles = particles[..., :-1]  # remove obj_on from features
            state_mask = torch.where(particles_obj_on.unsqueeze(-1) < 0, True, False).transpose(1, 2)

        # project particle features
        particles = self.particle_projection(particles)
        particles = particles.squeeze(1)

        if self.use_background:
            background_particle = self.background_projection(bg)
            particles = torch.cat([particles, background_particle.unsqueeze(1)], dim=1) # [bs, n_particles + 1, embed_dim]

        # forward through self-attention block1
        x = particles.unsqueeze(2)  # [bs, n_particles + 1, 1, embed_dim]
        if return_attention:
            x, attention_matrix = self.particle_self_att1(x, x_mask=state_mask, return_attention=True)
            attention_dict["self_1"] = attention_matrix
        else:
            x = self.particle_self_att1(x, x_mask=state_mask)

        # forward through self-attention block2
        if return_attention:
            x, attention_matrix = self.particle_self_att2(x, x_mask=state_mask, return_attention=True)
            attention_dict["self_2"] = attention_matrix
        else:
            x = self.particle_self_att2(x, x_mask=state_mask)

        # pool using special output particle
        out_particle = self.out_particle.repeat(bs, 1, 1)
        out_particle = out_particle.unsqueeze(2)  # [bs, 1, 1, embed_dim]
        if return_attention:
            x_agg, attention_matrix = self.particle_pool_att(out_particle, x, x_mask=state_mask, return_attention=True)
            attention_dict["agg"] = attention_matrix
        else:
            x_agg = self.particle_pool_att(out_particle, x, x_mask=state_mask)
        x_agg = x_agg.squeeze(1, 2)  # [bs, embed_dim]
        # final layer norm
        x_agg = self.linear_out(self.ln(x_agg))

        # forward through output MLP
        action = self.output_mlp(x_agg)  # [bs, 2 * action_dim]

        if return_attention:
            return action, attention_dict
        else:
            return action


class EITCriticNetwork(nn.Module):
    def __init__(self, particle_fdim, action_dim, embed_dim=64, h_dim=128, n_head=1, dropout=0.0, masking=False,
                 action_particle=True, out_dim=1, background_fdim=None, use_background=False):
        super().__init__()
        self.masking = masking
        self.action_particle = action_particle
        self.out_dim = out_dim
        self.action_projection = nn.Sequential(nn.Linear(action_dim, h_dim),
                                               nn.ReLU(True),
                                               nn.Linear(h_dim, embed_dim))
        self.background_dim = background_fdim
        self.use_background = use_background
        if self.use_background:
            self.background_projection = nn.Linear(background_fdim, embed_dim)

        particle_dim = particle_fdim - 1 if self.masking else particle_fdim
        self.particle_projection = nn.Linear(particle_dim, embed_dim)
        self.particle_self_att1 = EITBlock(embed_dim, h_dim, n_head,
                                           attn_pdrop=dropout, resid_pdrop=dropout, att_type='self')

        self.particle_self_att2 = EITBlock(embed_dim, h_dim, n_head,
                                           attn_pdrop=dropout, resid_pdrop=dropout, att_type='self')

        self.particle_pool_att = EITBlock(embed_dim, h_dim, n_head,
                                          attn_pdrop=dropout, resid_pdrop=dropout, att_type='cross')

        self.ln = nn.LayerNorm(embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim, bias=True)

        self.output_mlp = nn.Sequential(nn.Linear(2 * embed_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, self.out_dim))

        # special particle
        self.out_particle = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

    def forward(self, fg, action, bg=None):
        particles = fg
        bs, n_particles, feature_dim = particles.shape

        # preprocess particles and produce masks
        state_mask, goal_mask = None, None
        if self.masking:
            # prepare attention masks (based on obj_on)
            particles_obj_on = particles[..., -1].view(bs, -1)
            new_particles_obj_on = [particles_obj_on]
            if self.action_particle:
                new_particles_obj_on.insert(0, particles_obj_on.new_ones([bs, 1]))

            if self.use_background:
                new_particles_obj_on.append(particles_obj_on.new_ones([bs, 1]))

            if len(new_particles_obj_on) > 1:
                particles_obj_on = torch.cat(new_particles_obj_on, dim=-1)  # add special particles

            particles = particles[..., :-1]  # remove obj_on from features
            state_mask = torch.where(particles_obj_on.unsqueeze(-1) < 0, True, False).transpose(1, 2)

        # project particle features
        particles = self.particle_projection(particles)
        particles = particles.squeeze(1)

        # project action and add to particles
        action_particle = self.action_projection(action)
        x = [particles]
        if self.action_particle:
            x.insert(0, action_particle.unsqueeze(1))

        if self.use_background:
            background_particle = self.background_projection(bg)
            x.append(background_particle.unsqueeze(1))

        if len(x) == 1:
            x = particles  # [bs, n_particles, embed_dim]
        else:
            x = torch.cat(x, dim=1) # [bs, n_particles + 1, embed_dim] or [bs, n_particles + 2, embed_dim]

        # forward through self-attention block1
        x = x.unsqueeze(2)  # [bs, n_particles + ?, 1, embed_dim]
        x = self.particle_self_att1(x, x_mask=state_mask)

        # forward through self-attention block2
        x = self.particle_self_att2(x, x_mask=state_mask)

        # pool using special output particle
        out_particle = self.out_particle.repeat(bs, 1, 1)
        if self.action_particle:
            action_particle_out = x[:, 0].clone()
            x_out = torch.cat([out_particle, action_particle_out], dim=1)  # [bs, 2, embed_dim]
        else:
            x_out = out_particle
        x_out = x_out.unsqueeze(2)  # [bs, 2, 1, embed_dim]
        x_out = self.particle_pool_att(x_out, x, x_mask=state_mask)
        x_out = x_out.squeeze(2)  # [bs, 2, embed_dim]
        # final layer norm
        x_out = self.linear_out(self.ln(x_out))

        if self.action_particle:
            x_agg = torch.cat([x_out[:, 0], x_out[:, 1]], dim=-1)  # [bs, 2 * embed_dim]
        else:
            x_agg = torch.cat([x_out[:, 0], action_particle], dim=-1)  # [bs, 2 * embed_dim]

        # forward through output MLP
        output = self.output_mlp(x_agg)  # [bs, output_dim]
        return output


class EITCritic(nn.Module):
    def __init__(self, cfg, particle_fdim, action_dim, n_critics, out_dim, background_fdim):
        super().__init__()
        embed_dim = cfg.eit_embed_dim
        h_dim = cfg.eit_h_dim
        n_head = cfg.eit_n_head
        dropout = cfg.eit_dropout
        action_particle = cfg.eit_action_particle
        masking = cfg.eit_masking
        use_background = cfg.eit_use_background

        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = EITCriticNetwork(particle_fdim, action_dim, embed_dim, h_dim, n_head, dropout, masking,
                                     action_particle, out_dim, background_fdim, use_background)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, bg: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        qvalue_outputs = []
        for i in range(self.n_critics):
            value = self.q_networks[i](obs, actions, bg)
            qvalue_outputs.append(value)
        return tuple(qvalue_outputs)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor, bg: torch.Tensor = None) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        value = self.q_networks[0](obs, actions, bg)
        return value


if __name__ == '__main__':
    config = SimpleNamespace(**{
        'eit_embed_dim': 12, 'eit_h_dim': 11, 'eit_n_head': 3, 'eit_dropout': 0.1, 'eit_action_particle': True,
        'eit_masking': True, 'eit_n_critics': 2
    })
    particle_features_dim = 8
    act_dim = 3
    n_particles = 4

    critic = EITCritic(config, particle_features_dim, act_dim, n_critics=1, out_dim=1)
    observations = torch.zeros(1, n_particles, particle_features_dim, dtype=torch.float32)
    act = torch.zeros(1, act_dim, dtype=torch.float32)
    critic_output = critic(observations, act)
    print(critic_output)

    actor = EITActor(config, particle_features_dim, act_dim)
    actor_output = actor(observations)
    print(actor_output)
