import copy
from typing import Dict, List, Any

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from salsa.core.actions.actionMapper import ActionMapper
from salsa.entities.cluster import Cluster
from salsa.utils.typing import agent_id, cluster_id
from salsa.core.agents.nets.actorNetwork import ActorDeepSet
from salsa.core.agents.nets.criticNetwork import CriticDeepSet


class SalsaAgent:
    def __init__(self, aid: agent_id, cluster: Cluster, dims: Dict[str, int], action_dim, device, actor_lr=1e-4, critic_lr=1e-3, load_path:str=None):
        self.aid: agent_id = aid
        self.cluster = cluster
        self.action_dim = action_dim
        self.device = device
        c_dim = dims["cluster_dim"]
        ms_dim = dims["ms_dim"]
        slo_dim = dims["slo_dim"]
        n_dim = dims["neighbor_dim"]

        load_path = load_path or None

        self.actor = ActorDeepSet(
            cluster_dim=c_dim, ms_dim=ms_dim, slo_dim=slo_dim,
            neighbor_dim=n_dim, action_dim=action_dim
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)

        self.critic = CriticDeepSet(
            cluster_dim=c_dim, ms_dim=ms_dim, slo_dim=slo_dim,
            neighbor_dim=n_dim,
            ego_action_dim=action_dim,
            neighbor_action_dim=action_dim
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)

        if load_path:
            self.load_checkpoint(load_path + f"_{self.aid}.pt")

    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _collate(self, obs_list: List[Any]) -> Dict[str, torch.Tensor]:
        if not obs_list:
            return {}

        dict_list = [obs.to_dict() for obs in obs_list]

        batch = {}
        keys = dict_list[0].keys()

        for k in keys:
            val_list = [d[k] for d in dict_list]

            tensor_val = torch.FloatTensor(np.array(val_list)).to(self.device)
            batch[k] = tensor_val

        return batch

    def update(self, state_dict, actions_dict,
               rewards_dict, next_state_dict,
               dones_batch, masks,
               next_masks, neighbor_agents,
               gamma=0.99,
               tau=0.01,
               entropy_coeff=0.01):

        my_state = self._collate(state_dict[self.cluster.id])
        my_next_state = self._collate(next_state_dict[self.cluster.id])

        my_action_idx = actions_dict[self.aid].to(self.device)
        my_reward = rewards_dict[self.cluster.id].to(self.device).unsqueeze(1)
        dones_batch = dones_batch.to(self.device)

        huge_neg = torch.tensor(-1e9, device=self.device)
        def safe_mask(mask_tensor):
            no_valid_actions = ~mask_tensor.any(dim=-1)
            if no_valid_actions.any():
                mask_tensor[no_valid_actions, 0] = True
            return mask_tensor

        my_mask = masks[self.cluster.id].to(self.device).bool()
        my_mask = safe_mask(my_mask)
        my_action_onehot = F.one_hot(my_action_idx.long().squeeze(-1), num_classes=self.action_dim).float()

        # ----------------------------
        # STEP 1: TRAIN CRITIC
        # ----------------------------
        with torch.no_grad():
            nbr_next_actions_list = []

            for n_id, n_agent in neighbor_agents.items():
                if n_id == self.aid: continue

                n_cluster_id = n_agent.cluster.id
                raw_n_next_state = next_state_dict[n_cluster_id]
                n_next_state = self._collate(raw_n_next_state)

                n_logits = n_agent.target_actor(**n_next_state)

                n_mask = masks[n_cluster_id].to(n_agent.device).bool()
                n_mask = safe_mask(n_mask)
                masked_logits = n_logits.masked_fill(~n_mask, huge_neg)

                n_act = self.gumbel_softmax(masked_logits, hard=True)
                nbr_next_actions_list.append(n_act)

            if nbr_next_actions_list:
                nbr_next_actions = torch.stack(nbr_next_actions_list, dim=1)
            else:
                nbr_next_actions = torch.empty(
                    my_action_onehot.shape[0], 0, self.action_dim,
                    device=self.device
                )

            req_size = my_next_state['neighbors'].shape[1]
            curr_size = nbr_next_actions.shape[1]
            if curr_size < req_size:
                pad = torch.zeros(
                    nbr_next_actions.shape[0], req_size - curr_size, self.action_dim,
                    device=self.device
                )
                nbr_next_actions = torch.cat([nbr_next_actions, pad], dim=1)

            my_next_logits = self.target_actor(**my_next_state)
            my_next_mask = next_masks[self.cluster.id].to(self.device).bool()
            my_next_mask = safe_mask(my_next_mask)
            my_next_masked_logits = my_next_logits.masked_fill(~my_next_mask, huge_neg)
            my_next_action = self.gumbel_softmax(my_next_masked_logits, hard=True)

            target_q_values = self.target_critic(
                **my_next_state,
                ego_action=my_next_action,
                neighbor_actions=nbr_next_actions
            )
            target_q = my_reward + (1 - dones_batch) * gamma * target_q_values

        nbr_hist_actions_list = []
        for n_id in neighbor_agents.keys():
            if n_id == self.aid: continue

            n_act_idx = actions_dict[n_id].to(self.device)
            n_act_oh = F.one_hot(n_act_idx.long().squeeze(-1), num_classes=self.action_dim).float()
            nbr_hist_actions_list.append(n_act_oh)

        if nbr_hist_actions_list:
            nbr_hist_actions = torch.stack(nbr_hist_actions_list, dim=1)
        else:
            nbr_hist_actions = torch.empty(
                my_action_onehot.shape[0], 0, self.action_dim,
                device=self.device
            )

        req_size = my_state['neighbors'].shape[1]
        curr_size = nbr_hist_actions.shape[1]
        if curr_size < req_size:
            pad = torch.zeros(
                nbr_hist_actions.shape[0], req_size - curr_size, self.action_dim,
                device=self.device
            )
            nbr_hist_actions = torch.cat([nbr_hist_actions, pad], dim=1)

        current_q = self.critic(
            **my_state,
            ego_action=my_action_onehot,
            neighbor_actions=nbr_hist_actions
        )

        td_errors = F.mse_loss(current_q, target_q, reduction='none')
        critic_loss = td_errors.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ----------------------------
        # STEP 2: TRAIN ACTOR
        # ----------------------------
        my_current_logits = self.actor(**my_state)
        masked_logits = my_current_logits.masked_fill(~my_mask, huge_neg)

        my_new_action = self.gumbel_softmax(masked_logits, temperature=0.1, hard=False)

        actor_q_values = self.critic(
            **my_state,
            ego_action=my_new_action,
            neighbor_actions=nbr_hist_actions.detach() if nbr_hist_actions is not None else None
        )

        probs = F.softmax(masked_logits, dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)

        entropy = -torch.sum(torch.nan_to_num(probs * log_probs), dim=1).mean()

        actor_loss = -actor_q_values.mean() - (entropy_coeff * entropy)


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ----------------------------
        # STEP 3: SOFT UPDATES
        # ----------------------------
        self.soft_update(self.critic, self.target_critic, tau)
        self.soft_update(self.actor, self.target_actor, tau)

        with torch.no_grad():
            valid_actions_avg = my_mask.sum(dim=1).float().mean().item()

        metrics = {
            "loss_critic": critic_loss.item(),
            "loss_actor": actor_loss.item(),
            "value_mean_q": current_q.mean().item(),
            "policy_entropy": entropy.item(),
            "value_target_mean": target_q.mean().item(),
            "debug_valid_actions": valid_actions_avg
        }

        return td_errors.detach().cpu().numpy(), metrics

    def forward(self, observation) -> torch.Tensor:
        """
        observation: AgentObservation object (from Env)
        """
        state_dict_numpy = observation.to_dict()

        state_tensor = {
            k: torch.FloatTensor(v).unsqueeze(0)
            for k, v in state_dict_numpy.items()
        }

        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(**state_tensor)
        self.actor.train()

        return logits

    def select_action(self, logits: torch.Tensor, eval: bool = False):
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        logits = torch.where(torch.isposinf(logits), torch.zeros_like(logits), logits)

        if eval:
            action_idx = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

            if torch.isnan(probs).any():
                probs = torch.ones_like(logits) / logits.shape[-1]

            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()

        return action_idx.item()

    def save_checkpoint(self, path: str):
        """
        Saves the current state of the agent, including:
        - Actor and Critic weights
        - Target Actor and Target Critic weights
        - Optimizer states (to resume training correctly)
        """
        checkpoint = {
            'aid': self.aid,
            # Networks
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            # Optimizers
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }

        torch.save(checkpoint, path)
        print(f"Agent {self.aid}: Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Loads a checkpoint.
        Note: map_location ensures we load to the correct device (CPU/GPU)
        regardless of where it was saved.
        """
        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
        else:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        # 1. Load Networks
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

        # 2. Load Optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        print(f"Agent {self.aid}: Checkpoint loaded from {path}")
