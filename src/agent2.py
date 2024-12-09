import torch
import numpy as np
from network import FeedForwardNN
from memory import PPOMemory
from torch.distributions.categorical import Categorical


class Agent:

    def __init__(
        self,
        HP,
        batch_size,
        n_epochs,
        alpha,
        gamma,
        gae_lambda,
        policy_clip,
    ):
        self.HP = HP
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.alpha = alpha
        self.actor = FeedForwardNN(11, 1, self.alpha, HP=self.HP)
        self.critic = FeedForwardNN(11, 1, self.alpha, actor=False, HP=self.HP)
        self.memory = PPOMemory(batch_size)
        self.n_epochs = n_epochs

    def remember(self, state, action, probs, vals, reward, temp, done, current_episode):
        self.memory.store_memory(
            state, action, probs, vals, reward, temp, done, current_episode
        )

    def remember_location(self, location, distance_matrix, idx):
        self.memory.store_location(location, idx)
        self.memory.store_matrix(distance_matrix, idx)

    def save_models(self, path):
        # Save the actor1 and actor2 models
        self.actor.save_network(path, suffix="actor")

        # Save the critic model
        self.critic.save_network(path, suffix="critic")

    def generate_batches_data(
        self,
        observation: list[int],
        locations: dict[tuple],
        matrix,
        temp: float,
    ):
        batch_data = []
        indices = []
        for i in range(len(observation)):  # observation is a list of integers
            for j in range(len(observation)):
                if (
                    i == j
                    or (i + 1) % len(observation) == j
                    or (i - 1) % len(observation) == j
                    or i >= j
                ):
                    continue
                indices.append([i, j])
                batch_data.append(
                    [
                        # locations[observation[i - 1]][0],
                        # locations[observation[i - 1]][1],
                        matrix[i - 1, i],
                        locations[observation[i]][0],
                        locations[observation[i]][1],
                        matrix[i, (i + 1) % (len(observation))],
                        # locations[observation[(i + 1) % (len(observation))]][0],
                        # locations[observation[(i + 1) % (len(observation))]][1],
                        # locations[observation[j - 1]][0],
                        # locations[observation[j - 1]][1],
                        matrix[j - 1, j],
                        locations[observation[j]][0],
                        locations[observation[j]][1],
                        matrix[j, (j + 1) % (len(observation))],
                        # locations[observation[(j + 1) % (len(observation))]][0],
                        # locations[observation[(j + 1) % (len(observation))]][1],
                        matrix[i, j],
                        sum(
                            [
                                matrix[i, k]
                                for k in range(
                                    (i + 1) % (len(observation)),
                                    (j + 1) % (len(observation)),
                                )
                            ]
                        ),
                        temp,
                    ]
                )

        # Convert batch_data to PyTorch tensor
        batch_data = torch.tensor(batch_data, dtype=torch.float32)
        return batch_data, indices

    def forward_model(
        self,
        data: torch.tensor,
        model: FeedForwardNN,
    ):
        # Process in mini-batches
        mini_batch_size = 64
        logits_list = []
        for start_idx in range(0, len(data), mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, len(data))
            mini_batch = data[start_idx:end_idx]
            logits = model(mini_batch)  # Ensure model supports batch processing
            logits_list.append(logits)

        # Concatenate all logits
        logits = torch.cat(logits_list, dim=0)
        logits = logits.view(-1)

        return logits

    def choose_action(self, observation: list[int], episode: int, temp: float):
        locations = self.memory.get_locations(episode)
        matrix = self.memory.get_matrix(episode)
        data, indices = self.generate_batches_data(observation, locations, matrix, temp)
        # Get probabilities from actor1
        logits = self.forward_model(data, self.actor)
        # Apply Softmax to get probability distribution
        probs = torch.softmax(logits, dim=0)  # Normalize over the batch dimension

        # Sample first node based on probabilities
        dist = Categorical(probs=probs)  # Sample from probs1
        node_idx = dist.sample()
        action = indices[node_idx]

        # Log probabilities for PPO
        log_probs = torch.squeeze(dist.log_prob(node_idx))

        # Compute value from critic
        value = self.forward_model(data, self.critic)
        value = torch.squeeze(value.sum())

        return action, log_probs, value

    def split_episodes(self, rewards, values, dones):
        """
        Split data into individual episodes based on the `dones` flags.
        """
        episodes_rewards, episodes_values, episodes_dones = [], [], []
        start = 0

        for i, done in enumerate(dones):
            if done:  # End of an episode
                episodes_rewards.append(rewards[start : i + 1])
                episodes_values.append(values[start : i + 1])
                episodes_dones.append(dones[start : i + 1])
                start = i + 1

        return episodes_rewards, episodes_values, episodes_dones

    def calculate_gae(self, rewards, values, dones):
        """
        Calculate Generalized Advantage Estimation (GAE) for multiple episodes.
        """
        all_advantages = []

        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            # Backward pass for GAE calculation
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = (
                        ep_rews[t]
                        + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
                        - ep_vals[t]
                    )
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = (
                    delta
                    + self.gamma * self.gae_lambda * (1 - ep_dones[t]) * last_advantage
                )
                last_advantage = advantage
                advantages.insert(0, advantage)

            all_advantages.extend(advantages)  # Concatenate advantages

        return torch.tensor(all_advantages, dtype=torch.float32)

    def calculate_probability_ratio(self, old_policy_probs, states, episode, temp):
        states = states.tolist()
        new_probs = []
        critic_value = []
        for s, e, t in zip(states, episode, temp):
            _, new_prob, new_value = self.choose_action(
                s, e, t
            )  # new_prob ->prob, new_value -> value critic
            new_probs.append(new_prob)
            critic_value.append(new_value)
        new_probs = torch.stack(new_probs).to(self.actor.device)
        critic_value = torch.stack(critic_value).to(self.actor.device)
        prob_ratio = torch.exp(new_probs - old_policy_probs)
        return prob_ratio, critic_value

    def update_lr(self):
        self.actor.scheduler.step()
        self.critic.scheduler.step()

    def learn(self):
        batch_loss = []
        batch_actor_losses = []
        batch_critic_losses = []
        for _ in range(self.n_epochs):
            tot_losses = []
            actor_losses = []
            critic_losses = []
            # Generate mini-batches
            (
                state_arr,  # States observed during interaction
                action_arr,  # Actions taken
                old_prob_arr,  # Log-probabilities of the action
                # (from the policy at the time of collection)
                vals_arr,  # Value function estimates at the time of collection
                reward_arr,  # Rewards received
                temp_arr,  # Temperature values
                dones_arr,  # Flags indicating episode termination
                episodes,  # Episode indices for tracking
                batches,  # Mini-batches indices for training
            ) = self.memory.generate_batches()

            vals_arr = (
                vals_arr.clone().detach().to(torch.float32).to(self.actor.device)
            )  # Convert values to tensor in float32

            old_prob_arr = (
                old_prob_arr.clone().detach().to(torch.float32).to(self.actor.device)
            )  # Convert prob to tensor in float32

            # Split into episodes
            episodes_rewards, episodes_values, episodes_dones = self.split_episodes(
                reward_arr, vals_arr, dones_arr
            )

            # Calculate advantages
            advantages = self.calculate_gae(
                episodes_rewards, episodes_values, episodes_dones
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            # Calculate Probability Ratio
            for batch in batches:
                prob_ratio, critic_value = self.calculate_probability_ratio(
                    old_prob_arr[batch],
                    state_arr[batch],
                    episodes[batch],
                    temp_arr[batch],
                )
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self. policy_clip)
                    * advantages[batch]
                )
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Ctiric loss
                returns = advantages[batch] + vals_arr[batch]  # R_t = A_t + V(s)
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                # Backpropagation and optimization step
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                # Apply gradient clipping (e.g., max norm of 0.5)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                # Store losses for logging
                tot_losses.append(total_loss.item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
            batch_loss.append(np.mean(tot_losses))
            batch_actor_losses.append(np.mean(actor_losses))
            batch_critic_losses.append(np.mean(critic_losses))
        # Clear the memory after each optimization step
        self.memory.clear_memory()
        # self.update_lr()
        return (
            batch_loss,
            batch_actor_losses,
            batch_critic_losses,
        )
