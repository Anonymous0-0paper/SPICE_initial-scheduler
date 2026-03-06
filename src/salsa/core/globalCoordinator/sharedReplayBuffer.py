from collections import deque
from typing import List, Dict, Any

import numpy as np
import torch

from salsa.core.states.agentObservation import AgentObservation
from salsa.utils.typing import cluster_id, agent_id


class Transition:
    def __init__(self, state, actions, rewards, next_state, done, action_masks, next_action_masks, priority=1.0):
        self.state = state
        self.actions = actions
        self.rewards = rewards
        self.next_state = next_state
        self.done = done
        self.action_masks = action_masks
        self.next_action_masks = next_action_masks
        self.priority = priority


class SharedReplayBuffer:
    def __init__(self, config, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = config['scheduler']['replay_buffer']['batch_size']
        self.alpha = 0.6
        self.priorities = deque(maxlen=capacity)
        self.lambda_prio = config['scheduler']['replay_buffer']['lambda_prio']

    def push(self, state: Dict[cluster_id, AgentObservation],
             actions: Dict[agent_id, int],
             rewards: Dict[cluster_id, float],
             next_state: Dict[cluster_id, AgentObservation],
             done: bool, action_masks: Dict[cluster_id, List[bool]],
             next_action_masks: Dict[cluster_id, List[bool]],
             slo_violation_occurred=False):
        current_max = max(self.priorities) if self.priorities else 1.0
        initial_priority = current_max + self.lambda_prio if slo_violation_occurred else current_max

        t = Transition(state, actions, rewards, next_state, done, action_masks, next_action_masks, initial_priority)
        self.buffer.append(t)
        self.priorities.append(initial_priority)

    def clear(self):
        self.buffer.clear()
        self.priorities.clear()

    def sample(self):
        n = len(self.buffer)
        if n == 0: return None

        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, self.batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]

        state_list = [t.state for t in batch]
        action_list = [t.actions for t in batch]
        reward_list = [t.rewards for t in batch]
        next_state_list = [t.next_state for t in batch]
        done_list = [t.done for t in batch]
        mask_list = [t.action_masks for t in batch]
        next_mask_list = [t.next_action_masks for t in batch]

        state_batch = self._stack_dicts(state_list)
        next_state_batch = self._stack_dicts(next_state_list)
        action_batch = self._stack_dicts(action_list)
        reward_batch = self._stack_dicts(reward_list)
        mask_batch = self._stack_dicts(mask_list)
        next_mask_batch = self._stack_dicts(next_mask_list)

        done_batch = torch.FloatTensor(np.array(done_list)).unsqueeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, mask_batch, next_mask_batch, indices

    def update_priorities(self, indices: List[int], errors: np.ndarray):
        for i, err in zip(indices, errors):
            self.priorities[i] = (np.abs(err) + 1e-6) ** self.alpha

    def _stack_dicts(self, list_of_dicts: List[Dict]) -> Dict[str, Any]:
        """
        Collates a list of dictionaries (one per sample) into a single dictionary
        where values are batched tensors.
        """
        if not list_of_dicts: return {}

        batch_dict = {}
        keys = list_of_dicts[0].keys()

        for key in keys:
            val_list = [d[key] for d in list_of_dicts]
            first_elem = val_list[0]

            if isinstance(first_elem, (int, np.int32, np.int64)):
                val_array = np.array(val_list, dtype=np.int64)
                batch_dict[key] = torch.LongTensor(val_array)

            elif isinstance(first_elem, (float, np.float32, np.float64)):
                val_array = np.array(val_list, dtype=np.float32)
                batch_dict[key] = torch.FloatTensor(val_array)

            elif isinstance(first_elem, (torch.Tensor, np.ndarray)):
                batch_dict[key] = torch.tensor(np.array(val_list), dtype=torch.float32)

            elif isinstance(first_elem, list):
                val_array = np.array(val_list, dtype=np.float32)
                batch_dict[key] = torch.FloatTensor(val_array)

            else:
                batch_dict[key] = val_list

        return batch_dict

    def __len__(self):
        return len(self.buffer)