# Code adapted from PPL-MCTS (https://github.com/NohTow/PPL-MCTS)

from collections import defaultdict
import logging
from tqdm import tqdm
import time
import warnings
import numpy as np
import torch

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

def pad_sequences_to_left(sequences, batch_first=False, padding_value=0):
    """Add left padding so sequences have same shape"""
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, max_len-length:, ...] = tensor
        else:
            out_tensor[max_len-length:, i, ...] = tensor
    return out_tensor

def pad_sequences_to_left_states(sequences, padding_value=0, max_len=0):
    """Similar to pad_sequences_to_left function, but working on states tensor (in order to forge state for "sequential generation")"""
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    out_dims = (max_size[0], max_size[1], len(sequences), max_size[2], max_len, max_size[4])
    # print(out_dims)
    out_tensor = sequences[0].new_full(out_dims, padding_value, device=sequences[0].device)
    for i, tensor in enumerate(sequences):
        length = tensor.size()[3]
        out_tensor[:, :, i, :, max_len-length:, ...] = tensor
    return out_tensor

class PPO_MCTS:
    def __init__(self):
        pass

    def generate(
        self,
        input_ids, attention_mask,
        tokenizer, policy, value_model, ref_policy=None, reward_model=None,
        max_new_tokens=20, sim=10, k=10, c_puct=8.0, te=1.0, do_sample=False, td=1.0, anneal_td=True, top_p=1.0,
        gamma=1.0, kl_coef=0.0, clamp_kl=False, reward_gain=1.0, reward_bias=0.0,
        use_cache=True, log_level='ERROR',
    ):
        '''Entry point for PPO-MCTS generation
        Input:
        - input_ids: tensor(B, PL). A batch of tokenized prompts
        - attention_mask: tensor(B, L). The attention mask on the prompt tokens
        - Model parameters
            - policy: The PPO policy model (must be decoder-only Transformer)
            - value_model: The PPO value model (must be decoder-only Transformer)
            - ref_policy: [Optional] The reference policy model used for PPO training. If None, then we will "Approximate Q with V" (see paper Appendix A.4)
            - reward_model: [Optional] The reward model used for PPO training. If None, then we will "Approximate r(s_{T+1}) with V_\phi(s_{T+1})" (see paper Appendix A.4)
        - Generation parameters
            - max_new_tokens: int. Maximum number of tokens to generate
            - sim: int. Number of simulations to run per token
            - k: int. Number of top-k children actions to consider for each state
            - c_puct: float. Exploration constant for PUCT
            - te: float. Temperature applied to policy priors
            - do_sample: bool. Whether to sample from the visit counts
            - td: float. Temperature applied to the visit counts
            - anneal_td: bool. Whether to anneal td from 1.0 to 0.0 over the course of generation
            - top_p: float. Top-p sampling parameter applied to the visit counts
        - PPO parameters
            - gamma: float. Discount factor
            - kl_coef: float. KL coefficient. Useful only when ref_policy is provided
            - clamp_kl: bool. Whether to clamp the KL divergence. Useful only when kl_coef > 0.0
            - reward_gain: float. Gain for the reward model. Useful only when reward_model is provided
            - reward_bias: float. Bias for the reward model. Useful only when reward_model is provided
        - Misc parameters
            - use_cache: bool. Whether to use KV cache to save computation
            - log_level: One of { 'ERROR', 'WARNING', 'INFO', 'DEBUG' }
        Output:
        - input_ids: tensor(B, PL + CL). An expanded version of input_ids containing the generated tokens
        - attention_mask: tensor(B, PL + CL). The corresponding expanded attention mask
        '''

        self._batch_size = len(input_ids)
        self._batch_range = np.arange(self._batch_size)
        self._init_tree(
            tokenizer, policy, value_model, ref_policy, reward_model,
            max_new_tokens, sim, k, c_puct,
            te, do_sample, td, anneal_td, top_p,
            gamma, kl_coef, clamp_kl, reward_gain, reward_bias,
            use_cache,
            log_level,
        )

        original_len = len(input_ids[0])
        unfinished_sequences = torch.ones_like(input_ids[:, -1])
        for self._token_ix in tqdm(list(range(self._max_new_tokens))) if self._log_level in ['WARNING', 'INFO', 'DEBUG'] else range(self._max_new_tokens):
            if not unfinished_sequences.any():
                break

            # Logging for pre-token
            logger.info('================================')
            logger.info(f'Decoding token #{self._token_ix} ...')
            printed_prompt = self._tokenizer.decode(input_ids[0, :original_len], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            logger.info(f'Prompt: "{printed_prompt}"')
            printed_response = self._tokenizer.decode(input_ids[0, original_len:], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            logger.info(f'Existing response: "{printed_response}"')
            logger.info(f'')
            start_time = time.time()

            # Search
            res_search = self.search(input_ids, attention_mask)
            new_token_ids = torch.tensor(self.sample(res_search), dtype=torch.long, device=input_ids.device)
            new_token_ids = new_token_ids * unfinished_sequences + self._tokenizer.pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, torch.unsqueeze(new_token_ids, dim=1)], dim=1)
            attention_mask = torch.cat((attention_mask, torch.unsqueeze(unfinished_sequences, dim=1)), dim=1)
            unfinished_sequences = unfinished_sequences * (new_token_ids != self._tokenizer.eos_token_id).long()

            # Logging for post-token
            logger.info('----------------')
            logger.info(f'Decoded token #{self._token_ix}; New token id: {new_token_ids[0]}; New token: "{self._tokenizer.convert_ids_to_tokens(new_token_ids[0].item())}"')
            printed_prompt = self._tokenizer.decode(input_ids[0, :original_len], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            logger.info(f'Prompt: "{printed_prompt}"')
            printed_response = self._tokenizer.decode(input_ids[0, original_len:], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            logger.info(f'Response so far: "{printed_response}"')
            end_time = time.time()
            logger.info(f'Token #{self._token_ix} took {end_time - start_time} seconds.')
            logger.info('')
        return input_ids, attention_mask

    def _init_tree(
        self,
        tokenizer, policy, value_model, ref_policy, reward_model,
        max_new_tokens, sim, k, c_puct, te, do_sample, td, anneal_td, top_p,
        gamma, kl_coef, clamp_kl, reward_gain, reward_bias,
        use_cache, log_level,
    ):
        assert tokenizer is not None, 'tokenizer must be provided'
        self._tokenizer = tokenizer
        assert policy is not None, 'policy must be provided'
        self._policy = policy
        assert value_model is not None, 'value_model must be provided'
        self._value_model = value_model
        self._ref_policy = ref_policy
        self._reward_model = reward_model

        assert type(max_new_tokens) == int and max_new_tokens >= 1, f'max_new_tokens must be positive integer, got {max_new_tokens}'
        self._max_new_tokens = max_new_tokens
        assert type(sim) == int and sim >= 1, f'sim must be positive integer, got {sim}'
        self._sim = sim
        assert type(k) == int and k >= 1, f'k must be positive integer, got {k}'
        self._num_actions = tokenizer.vocab_size + 1
        self._k = min(k, self._num_actions)
        assert c_puct > 0.0, f'c_puct must be positive real number, got {c_puct}'
        self._c_puct = c_puct
        assert te > 0.0, f'te must be positive real number, got {te}'
        self._te = te
        assert type(do_sample) == bool, f'do_sample must be boolean, got {do_sample}'
        self._do_sample = do_sample
        assert td > 0.0, f'td must be positive real number, got {td}'
        self._td = td
        assert type(anneal_td) == bool, f'anneal_td must be boolean, got {anneal_td}'
        self._anneal_td = anneal_td
        assert top_p > 0.0, f'top_p must be positive real number, got {top_p}'
        self._top_p = top_p

        assert 0.0 < gamma <= 1.0, f'gamma must be real number in (0, 1], got {gamma}'
        self._gamma = gamma
        assert kl_coef >= 0.0, f'kl_coef must be non-negative real number, got {kl_coef}'
        self._kl_coef = kl_coef
        assert type(clamp_kl) == bool, f'clamp_kl must be boolean, got {clamp_kl}'
        self._clamp_kl = clamp_kl
        assert reward_gain > 0.0, f'reward_gain must be positive real number, got {reward_gain}'
        self._reward_gain = reward_gain
        assert type(reward_bias) == float, f'reward_bias must be real number, got {reward_bias}'
        self._reward_bias = reward_bias

        assert type(use_cache) == bool, f'use_cache must be boolean, got {use_cache}'
        self._use_cache = use_cache
        assert log_level in ['ERROR', 'WARNING', 'INFO', 'DEBUG'], f'log_level must be one of {"ERROR", "WARNING", "INFO", "DEBUG"}, got {log_level}'
        self._log_level = log_level
        logger.setLevel(log_level)

        # Allocate all necessary storage.
        # For a given search associated to a batch-index, node i is the i-th node
        # to be expanded. Node 0 corresponds to the root node.
        num_nodes = sim + 1
        batch_node = (self._batch_size, num_nodes)
        self._num_nodes = num_nodes
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        # action_from_parents[b, i] is the action taken to reach node i.
        # Note that action_from_parents[b, 0] will remain -1, as we do not know,
        # when doing search from the root, what action led to the root.
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        # The 0-indexed depth of the node. The root is the only 0-depth node.
        # The depth of node i, is the depth of its parent + 1.
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=bool)

        # To avoid costly numpy ops, we store a sparse version of the actions.
        # We select the top k actions according to the policy, and keep a mapping
        # of indices from 0 to k-1 to the actual action indices in the
        # self._topk_mapping tensor.
        batch_node_action = (self._batch_size, num_nodes, self._k)
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        self._children_qs = np.zeros(batch_node_action, dtype=np.float32)
        self._children_klpens = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        self._states_by_model = defaultdict(dict) # mapping from (b, i) to the key-value state of the policy

        self._sim_ix = 0
        self._token_ids = {}
        self._attention_mask = {}

    def _reset_tree(self):
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)
        self._is_terminal.fill(False)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_qs.fill(0.0)
        self._children_klpens.fill(0.0)
        self._children_visits.fill(0)
        self._states_by_model = defaultdict(dict)

        self._sim_ix = 0
        self._token_ids = {}
        self._attention_mask = {}

    def sample(self, res_search):
        if not self._do_sample:
            return np.argmax(res_search, axis=1)
        else:
            # normalize along axis=1
            probs = res_search / np.sum(res_search, axis=1, keepdims=True)
            if self._top_p != 1.0:
                # sort along axis=1, descending; use argsort
                sorted_indices = np.argsort(-probs, axis=1)
                sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
                # compute cumulative sum along axis=1
                cumprobs = np.cumsum(sorted_probs, axis=1)
                # find the first index where cumprobs > top_p
                topp_indices = np.argmax(cumprobs > self._top_p, axis=1)
                # zero out indices after topp_indices
                for i in range(self._batch_size):
                    sorted_probs[:, (topp_indices[i] + 1):] = 0.0
                # re-normalize along axis=1
                sorted_probs = sorted_probs / np.sum(sorted_probs, axis=1, keepdims=True)
                # restore original order
                probs = np.take_along_axis(sorted_probs, np.argsort(sorted_indices, axis=1), axis=1)
            # sample from probs
            return np.array([np.random.choice(self._num_actions, p=probs[i]) for i in range(self._batch_size)])

    def search(self, input_ids, attention_mask):
        self._reset_tree()

        # Evaluate the root
        evaluation = self.evaluate(input_ids=input_ids, attention_mask=attention_mask)
        root_index = 0
        self.create_node(root_index, evaluation, input_ids, attention_mask, np.full(self._batch_size, False, dtype=bool))

        # Do simulations, expansions, and backwards
        leaf_indices = np.zeros((self._batch_size), np.int32)
        for self._sim_ix in range(self._sim):
            logger.debug('----------------')
            logger.debug(f"Simulation #{self._sim_ix} ...")
            start_time = time.time()
            self.print_tree()

            node_indices, actions = self.select()
            next_node_index = self._sim_ix + 1 # root is 0, therefore we offset by 1.
            self.expand(node_indices, actions, next_node_index)
            leaf_indices.fill(next_node_index)
            self.backward(leaf_indices)

            end_time = time.time()
            logger.debug(f'Simulation #{self._sim_ix} took {(end_time - start_time):.4f} seconds')
            logger.debug('')

        # return visit counts of children of root
        return self.dense_visit_counts()

    def select(self):
        logger.debug('Selecting ...')

        node_indices = np.zeros((self._batch_size), np.int32)
        depth = 0
        highlight_nodes = [] # [i]
        highlight_edges = [] # [(i, a)]
        while True:
            logger.debug(f'\tdepth={depth}, node_index={node_indices[0]}')
            depth += 1
            actions = self.uct_select_action(node_indices)
            logger.debug(f'\tselected_action={actions[0]}')
            next_node_indices = self._children_index[self._batch_range, node_indices, actions]
            highlight_nodes.append(node_indices[0])
            highlight_edges.append((node_indices[0], actions[0]))
            is_unexplored = next_node_indices == -1
            if is_unexplored.all():
                break
            else:
                node_indices = np.where(is_unexplored, node_indices, next_node_indices)
        
        logger.debug(f'\tSelected node {node_indices[0]}, action {actions[0]}')
        return node_indices, actions

    def uct_select_action(self, node_indices):
        node_children_prior = self._children_prior[self._batch_range, node_indices, :] # (B, A)
        node_children_qs = self._children_qs[self._batch_range, node_indices, :] # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :] # (B, A)
        node_visits = self._visit_counts[self._batch_range, node_indices] # (B)
        node_policy_score = np.sqrt(node_visits[:, None]) * self._c_puct * node_children_prior / (node_children_visits + 1) # (B, A)

        node_value_score = node_children_qs

        node_uct_score = node_value_score + node_policy_score # (B, A)

        for a in range(self._k):
            logger.debug(f'\t\taction {a}: policy_score={node_policy_score[0, a]:.4f}, value_score={node_value_score[0, a]:.4f}, uct_score={node_uct_score[0, a]:.4f}')
        actions = np.argmax(node_uct_score, axis=1)
        return actions

    def expand(self, node_indices, actions, next_node_index):
        logger.debug('Expanding ...')

        # Retrieve token ids for nodes to be evaluated.
        tokens_ids = pad_sequences_to_left([self._token_ids[(b, n)] for b, n in enumerate(node_indices)], True, self._tokenizer.eos_token_id)
        attention_masks = pad_sequences_to_left([self._attention_mask[(b, n)] for b, n in enumerate(node_indices)], True, 0)

        # Load states
        states_by_model = self.load_states(node_indices, max_len=len(tokens_ids[0]))

        # Convert sparse actions to dense actions for network computation
        dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
        # Add actions to list of tokens and extend attention mask by 1
        tokens_ids = torch.cat((tokens_ids, torch.unsqueeze(torch.cuda.LongTensor(dense_actions), 1)), dim = 1)
        attention_masks = torch.cat((attention_masks, torch.unsqueeze(torch.ones(len(dense_actions), dtype=torch.long, device=attention_masks.device), 1)), dim = 1)

        # Check if expanded nodes are terminal
        expanded_node_is_terminal = dense_actions == self._tokenizer.eos_token_id

        # Evaluate nodes
        evaluation = self.evaluate(input_ids=tokens_ids, attention_mask=attention_masks, states_by_model=states_by_model)

        # Create the new nodes
        self.create_node(next_node_index, evaluation, tokens_ids, attention_masks, expanded_node_is_terminal)

        # Update tree topology
        self._children_index[self._batch_range, node_indices, actions] = next_node_index
        self._parents[:, next_node_index] = node_indices
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range, node_indices] + 1

        logger.debug(f'\tCreated node {next_node_index}: n={self._visit_counts[0, next_node_index]}, v={self._values[0, next_node_index]:.4f}, parent={self._parents[0, next_node_index]}, action_from_parent={self._action_from_parents[0, next_node_index]}, depth={self._depth[0, next_node_index]}, is_terminal={self._is_terminal[0, next_node_index]}')
        for a in range(self._k):
            logger.debug(f'\t\taction {a}: token_id={self._topk_mapping[0, next_node_index, a]}, child_index={self._children_index[0, next_node_index, a]}, child_p={self._children_prior[0, next_node_index, a]:.4f}, child_q={self._children_qs[0, next_node_index, a]:.4f}, child_klpen={self._children_klpens[0, next_node_index, a]:.4f}, child_n={self._children_visits[0, next_node_index, a]}')

    def evaluate(self, input_ids, attention_mask, states_by_model=None):
        '''Get score from current nodes
        Inputs:
        - input_ids: (B, L), including the new node's token
        - attention_mask: (B, L)
        - states_by_model (nullable)
        Outputs:
        - priors: (B, V), numpy, on CPU, float32; the policy priors for the new node, corresponds to the P(s, a) in the AlphaGo paper
        - values: (B), numpy, on CPU, float32; this is the value of the new node
        - klpens (nullable): (B, V), numpy, on CPU, float32; the KL penalty for the new node's children
        - next_states_by_model (nullable): dict with keys 'policy', 'value', 'ref_policy' (optional), 'reward' (optional)
        '''

        next_states_by_model = {}

        policy_outputs = self._policy(**(self._policy.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask, past_key_values=states_by_model['policy'] if states_by_model is not None else None, use_cache=True)))
        priors = torch.nn.functional.softmax(policy_outputs.logits[:, -1, :] / self._te, dim=-1) # (B, V)
        next_states_by_model['policy'] = policy_outputs.past_key_values

        value_outputs = self._value_model(**(self._value_model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask, past_key_values=states_by_model['value'] if states_by_model is not None else None, use_cache=True)))
        values = value_outputs.logits[:, -1] # (B)
        next_states_by_model['value'] = value_outputs.past_key_values

        klpens = None
        if self._kl_coef != 0.0:
            ref_policy_outputs = self._ref_policy(**(self._ref_policy.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask, past_key_values=states_by_model['ref_policy'] if states_by_model is not None else None, use_cache=True)))
            ref_priors = torch.nn.functional.softmax(ref_policy_outputs.logits[:, -1, :] / self._te, dim=-1) # (B, V)
            entropy = priors - ref_priors
            if self._clamp_kl:
                entropy = torch.clamp(entropy, min=0.0)
            klpens = -self._kl_coef * entropy
            klpens = klpens.float().cpu().detach().numpy()
            next_states_by_model['ref_policy'] = ref_policy_outputs.past_key_values

            if self._reward_model is not None:
                reward_outputs = self._reward_model(**(self._reward_model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask, past_key_values=states_by_model['reward'] if states_by_model is not None and 'reward' in states_by_model else None, use_cache=True)))
                rewards = reward_outputs.logits[:, -1] # (B)
                rewards = rewards * self._reward_gain + self._reward_bias # normalize rewards
                terminal_mask = (input_ids[:, -1] == self._tokenizer.eos_token_id).long()
                values = values * (1 - terminal_mask) + rewards * terminal_mask
                if 'next_states' in reward_outputs:
                    next_states_by_model['reward'] = reward_outputs.past_key_values

        priors = priors.float().cpu().detach().numpy()
        values = values.float().cpu().detach().numpy()
        if not self._use_cache:
            next_states_by_model = None
        return dict(priors=priors, values=values, klpens=klpens, next_states_by_model=next_states_by_model)

    def backward(self, leaf_indices):
        logger.debug(f'Backward ...')

        node_indices = leaf_indices # (B)
        highlight_nodes = []
        highlight_edges = []
        while True:
            is_root = node_indices == 0
            if is_root.all():
                break
            parents = np.where(is_root, 0, self._parents[self._batch_range, node_indices])
            if parents[0] != -1:
                highlight_nodes.append(parents[0])
                a = self._action_from_parents[0, node_indices[0]]
                highlight_edges.append((parents[0], a))
            root_mask = 1.0 * is_root
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - root_mask
            parent_terminal_mask = 1.0 * self._is_terminal[self._batch_range, parents]
            root_or_parent_terminal_mask = 1.0 - (1.0 - root_mask) * (1.0 - parent_terminal_mask)
            # Update the parent nodes iff their child is not the root
            # We therefore mask the updates using not_root_mask and root_mask
            actions = np.where(is_root, 0, self._action_from_parents[self._batch_range, node_indices])
            old_qs = self._children_qs[self._batch_range, parents, actions].copy()
            if self._kl_coef != 0.0:
                self._children_qs[self._batch_range, parents, actions] = \
                    root_mask * self._children_qs[self._batch_range, parents, actions] + \
                    not_root_mask * (self._gamma * self._values[self._batch_range, node_indices] + self._children_klpens[self._batch_range, parents, actions])
            else: # self._kl_coef == 0.0
                self._children_qs[self._batch_range, parents, actions] = \
                    root_mask * self._children_qs[self._batch_range, parents, actions] + \
                    not_root_mask * self._values[self._batch_range, node_indices]
            self._children_visits[self._batch_range, parents, actions] += not_root_mask_int
            self._values[self._batch_range, parents] = \
                root_or_parent_terminal_mask * self._values[self._batch_range, parents] + \
                (1.0 - root_or_parent_terminal_mask) * \
                    (self._values[self._batch_range, parents] * self._visit_counts[self._batch_range, parents] \
                     - old_qs * (self._visit_counts[self._batch_range, node_indices] - 1) \
                     + self._children_qs[self._batch_range, parents, actions] * self._visit_counts[self._batch_range, node_indices]) \
                    / (self._visit_counts[self._batch_range, parents] + 1.0)
            self._visit_counts[self._batch_range, parents] += not_root_mask_int

            if not is_root[0]:
                logger.debug(f'\t\taction {actions[0]}: updated n={self._children_visits[0, parents[0], actions[0]]}, q={self._children_qs[0, parents[0], actions[0]]:.4f}')
                logger.debug(f'\tnode {parents[0]}: updated n={self._visit_counts[0, parents[0]]}, v={self._values[0, parents[0]]:.4f}')

            # Go up
            node_indices = parents

    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_visit_counts
        td = self._td if not self._anneal_td else (self._td * (1 - self._token_ix / self._max_new_tokens))
        dense_visit_counts = dense_visit_counts ** (1.0 / td)
        return dense_visit_counts

    def create_node(self, node_index, evaluation, tokens_ids, attention_masks, expanded_node_is_terminal):
        # Truncate the prior to only keep the top k logits
        prior, values, klpens, next_states_by_model = evaluation['priors'], evaluation['values'], evaluation['klpens'], evaluation['next_states_by_model']

        prior_topk_indices = np.argpartition(prior, -self._k, axis=-1)[:, -self._k:]
        prior = prior[self._batch_range[:, None], prior_topk_indices] # (B, A)
        if self._kl_coef != 0.0:
            klpens = klpens[self._batch_range[:, None], prior_topk_indices] # (B, A)

        # Store the indices of the top k logits
        self._topk_mapping[self._batch_range, node_index, :] = prior_topk_indices

        # Update prior, values and visit counts.
        self._children_prior[:, node_index, :] = prior
        if self._kl_coef != 0.0:
            self._children_klpens[:, node_index, :] = klpens
        self._values[:, node_index] = values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal

        # Initialize the children Qs with the parent value
        self._children_qs[:, node_index, :] = values[:, np.newaxis]

        # Save states
        self.save_states(node_index, next_states_by_model)

        # Updates tokens ids and attention masks
        for b, token_ids in enumerate(tokens_ids):
            self._token_ids[(b, node_index)] = token_ids
        for b, attention_mask in enumerate(attention_masks):
            self._attention_mask[(b, node_index)] = attention_mask

    def save_states(self, node_index, states_by_model):
        if not self._use_cache:
            return
        for model, states in states_by_model.items():
            # Transform the returned states format into tensor for easier manipulation
            key_value_tensor = torch.stack(list(
                torch.stack(list(
                    states[i]
                ), dim=0) for i in range(len(states))
            ), dim=0)  # (Y, 2, B, H, L, D/H)
            for b in range(states[0][0].size(0)):
                if node_index == 0:
                    self._states_by_model[model][(b, node_index)] = torch.clone(key_value_tensor[:, :, b])
                else:
                    self._states_by_model[model][(b, node_index)] = torch.clone(key_value_tensor[:, :, b, :, -1:])

    def get_states_from_node(self, b, n, model):
        state_array = []
        state_array.append(self._states_by_model[model][(b, n)])
        while n != 0:
            n = self._parents[(b, n)]
            state_array.append(self._states_by_model[model][(b, n)])
        state_array = state_array[::-1]
        result = torch.cat(state_array, 3)
        return result

    def load_states(self, node_indices, max_len):
        if not self._use_cache:
            return None
        states_by_model = {}
        for model in ['policy', 'value', 'ref_policy', 'reward']:
            try:
                states = [self.get_states_from_node(b, n, model) for b, n in enumerate(node_indices)]
            except:
                continue
            states_tensor = pad_sequences_to_left_states(states, padding_value=0, max_len=max_len)
            states = tuple(tuple(type_of_value for type_of_value in layer) for layer in states_tensor)
            states_by_model[model] = states
        return states_by_model

    def print_tree(self):
        logger.debug(f'Current tree:')
        for i in range(self._sim_ix + 1): # sim + 1 is the current number of nodes
            logger.debug(f'\tnode {i}: n={self._visit_counts[0, i]}, v={self._values[0, i]:.4f}, parent={self._parents[0, i]}, action_from_parent={self._action_from_parents[0, i]}, depth={self._depth[0, i]}, is_terminal={self._is_terminal[0, i]}')
            for a in range(self._k):
                logger.debug(f'\t\taction {a}: token_id={self._topk_mapping[0, i, a]}, child_index={self._children_index[0, i, a]}, child_p={self._children_prior[0, i, a]:.4f}, child_q={self._children_qs[0, i, a]:.4f}, child_klpen={self._children_klpens[0, i, a]:.4f}, child_n={self._children_visits[0, i, a]}')
