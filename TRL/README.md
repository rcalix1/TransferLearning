# TRL 

* using the older version of TRL
* python 3.8 or 3.9
* use older version of TRL
* pip install trl==0.11.3
* 

## Notebooks

* SFT
* RLHF - sentiment
* RLHF - spam (https://github.com/rcalix1/TransferLearning/blob/main/RLHF/ITS530-DavidHigley-gpt2-phish-spam.ipynb)
* Noah - Math GPT
* Fine Tune a BERT classifier
* GRPO (new TRL)
* 


```

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# === Tiny Policy Model (like babyGPT, no transformer) ===
class TinyPolicy(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 8)  # Simple embedding
        self.head = nn.Linear(8, 1)               # Output score for token

    def forward(self, token_ids):
        x = self.embed(token_ids)  # [batch, 8]
        return self.head(x).squeeze(-1)  # [batch] — logits for taken tokens


# === Reward function: +1 if token == 1 ===
def fake_reward(token_ids):
    return torch.where(token_ids == 1, torch.tensor(1.0), torch.tensor(0.0))


# === PPO Loss Function ===
def ppo_loss(new_logits, old_logits, actions, rewards, kl_coeff=0.1):
    new_logprobs = F.logsigmoid(new_logits)
    old_logprobs = F.logsigmoid(old_logits).detach()  # fixed baseline

    # Importance sampling ratio
    ratio = torch.exp(new_logprobs - old_logprobs)

    # PPO surrogate objective (no advantage, just raw reward)
    surrogate = ratio * rewards

    # KL divergence
    kl = (old_logprobs - new_logprobs).mean()

    # Final PPO loss
    return -surrogate.mean() + kl_coeff * kl


# === Training Loop ===
vocab_size = 10
policy = TinyPolicy(vocab_size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for step in range(300):
    # === Sample tokens ===
    logits = torch.randn(vocab_size)  # dummy logits for sampling space
    probs = F.softmax(logits, dim=0)  # [vocab]
    actions = torch.multinomial(probs, num_samples=8, replacement=True)  # [8 tokens]

    # === Old policy output (logits for those tokens) ===
    with torch.no_grad():
        old_logits = policy(actions)  # [8]

    # === Reward based on action taken ===
    rewards = fake_reward(actions)  # [8], +1 if token == 1 else 0

    # === New policy output ===
    new_logits = policy(actions)  # [8]

    # === Compute PPO loss ===
    loss = ppo_loss(new_logits, old_logits, actions, rewards)

    # === Optimize ===
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # === Print progress ===
    if step % 50 == 0:
        avg_reward = rewards.float().mean().item()
        print(f"Step {step} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.2f}")




```
