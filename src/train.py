import random
import time
from collections import deque, namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR, SequentialLR

from interface import Agent
from training_env import evaluate_agent, init, step


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, gamma, n_steps=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_steps_buffer = deque(maxlen=self.n_steps)

    def append(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        """Add a new experience to memory."""
        self.n_steps_buffer.append((state, action, reward, next_state, done))
        if len(self.n_steps_buffer) == self.n_steps:
            state, action, reward, next_state, done = self.calc_multistep_return()
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_steps):
            Return += self.gamma**idx * self.n_steps_buffer[idx][2]

        return (
            self.n_steps_buffer[0][0],
            self.n_steps_buffer[0][1],
            Return,
            self.n_steps_buffer[-1][3],
            self.n_steps_buffer[-1][4],
        )

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.stack([e.state for e in experiences if e is not None])
            .float()
            .to(self.device)
        )
        actions = (
            torch.vstack([e.action for e in experiences if e is not None])
            .long()
            .to(self.device)
        )
        rewards = (
            torch.vstack([e.reward for e in experiences if e is not None])
            .float()
            .to(self.device)
        )
        next_states = (
            torch.stack([e.next_state for e in experiences if e is not None])
            .float()
            .to(self.device)
        )
        dones = (
            torch.vstack([e.done for e in experiences if e is not None])
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def greedy_action(network, state):
    network.eval()
    device = "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class DQN(Agent):
    def __init__(
        self,
        nb_actions: int,
        nb_states: int,
        gamma: float = 0.95,
        batch_size: int = 100,
        buffer_size: int = int(1e5),
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay_period: int = 1000,
        epsilon_delay_decay: int = 20,
        learning_rate: float = 0.001,
        gradient_steps: int = 1,
        update_target_strategy: str = "ema",
        update_target_freq: int = 20,
        update_target_tau: float = 0.005,
        criterion: nn.modules.loss = torch.nn.MSELoss(),
        optimizer: torch.optim.Optimizer = None,
        n_steps: int = 1,
        compute_score_every: int = 5,
        hidden_size: int = 512,
        domain_randomization: bool = False,
        p: float = 0.5,
    ):
        """
        Args:
            nb_actions: Number of possible actions in the environment
            nb_states: Number of features in the state representation
            gamma: Discount factor for the reward in the Q-learning update
            batch_size: Size of the batch used for training
            buffer_size: Maximum size of the replay buffer
            epsilon_max: Initial epsilon value for the epsilon-greedy policy
            epsilon_min: Final epsilon value for the epsilon-greedy policy
            epsilon_decay_period: Number of steps to decay epsilon from epsilon_max to epsilon_min
            epsilon_delay_decay: Number of steps before starting to decay epsilon
            learning_rate: Learning rate for the optimizer
            gradient_steps: Number of gradient steps to perform at each interaction with the environment
            update_target_strategy: Strategy to update the target network. Either 'replace' or 'ema'
            update_target_freq: Frequency at which to update the target network
            update_target_tau: Interpolation factor for the EMA update
            criterion: Loss function to use
            optimizer: Optimizer to use
            n_steps: Number of steps to look ahead in the Q-learning update
            hidden_size: Number of units in the hidden layers
            domain_randomization: Whether to use domain randomization during training
            p: Probability of using the base environment when using domain randomization
        """

        self.nb_actions = nb_actions
        self.nb_states = nb_states
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_period = epsilon_decay_period
        self.epsilon_delay_decay = epsilon_delay_decay
        self.epsilon_step = (
            self.epsilon_max - self.epsilon_min
        ) / self.epsilon_decay_period
        self.learning_rate = learning_rate
        self.gradient_steps = gradient_steps
        self.update_target_strategy = update_target_strategy
        self.update_target_freq = update_target_freq
        self.update_target_tau = update_target_tau
        self.criterion = criterion
        self.n_steps = n_steps
        self.compute_score_every = compute_score_every
        self.hidden_size = hidden_size
        self.domain_randomization = domain_randomization
        self.p = p

        self.device = "cpu"

        self.model = torch.nn.Sequential(
            nn.Linear(self.nb_states, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.nb_actions),
        ).to(self.device)

        self.model.apply(self.weight_init)

        self.memory = ReplayBuffer(
            self.buffer_size, self.batch_size, self.device, self.gamma, self.n_steps
        )
        self.optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            if optimizer is None
            else optimizer
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: 1),
                MultiplicativeLR(
                    self.optimizer,
                    lr_lambda=lambda epoch: 0.99,
                ),
            ],
            milestones=[49],
        )
        self.target_model = deepcopy(self.model).to(self.device)

    def get_domain_randomization(self):
        # When training with domain randomization, we still want to train on the base environment some of the time (p=0.5 by default)
        if self.domain_randomization:
            return np.random.rand() > self.p
        else:
            return False

    def weight_init(self, layer: nn.Module):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)

    def gradient_step(self):
        # Double DQN
        self.model.train()
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample()
            current_max_action = self.model(Y).argmax(1).to(torch.long).unsqueeze(1)
            target_values = (
                self.target_model(Y)
                .detach()
                .gather(1, current_max_action)
                .reshape(self.batch_size, -1)
            )
            target = R + torch.mul(1 - D, self.gamma * target_values)
            estimation = (
                self.model(X).gather(1, A.to(torch.long)).reshape(self.batch_size, -1)
            )
            loss = self.criterion(estimation, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        else:
            return 0

    def train_without_interaction(self, epochs: int):
        """The training is very slow because of the environment interactions. While this is good for exploration, we can fine tune the model on the data we have collected so far, or we can use this for initial training."""
        nb_samples = len(self.memory)
        for e in range(epochs):
            loss = 0
            for _ in range(nb_samples // self.batch_size):
                loss += self.gradient_step()
            loss /= nb_samples // self.batch_size
            print(
                f"Epoch {e}, loss {loss:.2E}, lr {self.optimizer.param_groups[0]['lr']:.2E}, time {time.strftime('%H:%M:%S', time.gmtime())}"
            )

    def fill_memory(self, limit=None):
        state, params = init(domain_randomization=self.domain_randomization)
        iter = 0
        limit = limit if limit is not None else self.buffer_size
        while len(self.memory) < limit:
            iter += 1
            action = random.randint(0, self.nb_actions - 1)
            next_state, reward, done, _, _ = step(state, action, params)
            self.memory.append(
                torch.tensor(state, requires_grad=False),
                torch.tensor(action, requires_grad=False),
                torch.tensor(reward, requires_grad=False),
                torch.tensor(next_state, requires_grad=False),
                torch.tensor(done, requires_grad=False),
            )
            state = next_state
            if iter % 200 == 0:
                print(f"Memory {len(self.memory):4d}", end="\r")
                state, params = init(domain_randomization=self.domain_randomization)

    def train(self, episodes: int):
        self.fill_memory(limit=10000)
        self.train_without_interaction(200)
        try:
            rewards = []
            scores = []
            episode = 0
            iterations = 0
            cumulated_loss = 0
            cumulated_reward = 0
            best_score = 0
            best_episode_reward = 0
            epsilon = self.epsilon_max

            state, params = init(domain_randomization=self.get_domain_randomization())

            while episode < episodes:
                iterations += 1

                if iterations > self.epsilon_delay_decay:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

                # action selection
                if np.random.rand() < epsilon:
                    action = random.randint(0, self.nb_actions - 1)
                else:
                    action = greedy_action(self.model, state)

                # environment interaction
                next_state, reward, done, trunc, _ = step(state, action, params)
                if iterations % 200 == 0:
                    trunc = True

                self.memory.append(
                    torch.tensor(state, requires_grad=False),
                    torch.tensor(action, requires_grad=False),
                    torch.tensor(reward, requires_grad=False),
                    torch.tensor(next_state, requires_grad=False),
                    torch.tensor(done, requires_grad=False),
                )
                cumulated_reward += reward

                # gradient steps
                for _ in range(self.gradient_steps):
                    cumulated_loss += self.gradient_step()

                # update target network if needed
                if self.update_target_strategy == "replace":
                    if iterations % self.update_target_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                if self.update_target_strategy == "ema":
                    target_state_dict = self.target_model.state_dict()
                    model_state_dict = self.model.state_dict()
                    tau = self.update_target_tau
                    for key in model_state_dict:
                        target_state_dict[key] = (
                            tau * model_state_dict[key]
                            + (1 - tau) * target_state_dict[key]
                        )
                    self.target_model.load_state_dict(target_state_dict)

                # update state or reset environment
                if done or trunc:
                    if len(self.memory) > self.batch_size:
                        self.scheduler.step()
                    rewards.append(cumulated_reward)
                    if cumulated_reward > best_episode_reward:
                        best_episode_reward = cumulated_reward
                    if (
                        (episode % self.compute_score_every == 0)
                        or (best_episode_reward == cumulated_reward)
                        or (cumulated_reward > 1e10)
                    ):
                        score = evaluate_agent(agent=self)
                    else:
                        score = scores[-1]
                    scores.append(score)

                    episode += 1
                    print(
                        f"Episode {episode:2d}, epsilon {epsilon:6.2f}, memory {len(self.memory):4d}, episode reward {cumulated_reward:.2E}, loss {cumulated_loss:.2E}, lr {self.optimizer.param_groups[0]['lr']:.2E}, time {time.strftime('%H:%M:%S', time.gmtime())}, score {score:.2E}",
                    )

                    if score > best_score:
                        best_score = score
                        self.save("./model.pt")

                    state, params = init(
                        domain_randomization=self.get_domain_randomization()
                    )
                    cumulated_reward = 0
                    cumulated_loss = 0
                else:
                    state = next_state

            return rewards, scores

        except KeyboardInterrupt:
            return rewards, scores

    def act(self, observation, use_random=False):
        return greedy_action(self.model, observation)

    def save(self, path="./model.pt"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="./model.pt"):
        self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
