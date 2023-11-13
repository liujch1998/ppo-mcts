# PPO-MCTS

This repo hosts the code for [Don't throw away your value model! Making PPO even better via Value-Guided Monte-Carlo Tree Search decoding](https://arxiv.org/abs/2309.15028)

This repo has a text generation plugin that implements the Monte-Carlo Tree Search (MCTS) decoding on PPO-trained models.

## Usage

**Prerequisite:**
You need to be on PyTorch, and need to have an existing PPO-trained model that you intend to generate text from.
You need to have both the policy and value checkpoints, both must be decoder-only Transformers.
The value model should output a logit with shape `(batch_size, 1)`.

You can replace you usual generation call
```
policy.generate(input_ids, attention_mask)
```
with the following:
```
from ppo_mcts import PPO_MCTS
PPO_MCTS().generate(input_ids, attention_mask, tokenizer, policy, value_model)
```
Additional hyper-parameters are specified in `PPO_MCTS.generate()`.

## Citation

If you find this repo useful, please consider citing our paper:
```
@inproceedings{Liu2023DontTA,
  title={Don't throw away your value model! Making PPO even better via Value-Guided Monte-Carlo Tree Search decoding},
  author={Jiacheng Liu and Andrew Cohen and Ramakanth Pasunuru and Yejin Choi and Hannaneh Hajishirzi and Asli Celikyilmaz},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:262824527}
}
```
