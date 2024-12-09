import torch
import numpy as np
from network import FeedForwardNN, init_combined_parameters
from memory import PPOMemory
import os


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
        self.actor1 = FeedForwardNN(7, 1, self.alpha, HP=self.HP, use_optimizer=False)
        self.actor2 = FeedForwardNN(13, 1, self.alpha, HP=self.HP, use_optimizer=False)
        self.combined_optimizer, self.scheduler = init_combined_parameters(
            self.actor1, self.actor2, self.alpha
        )
        self.critic = FeedForwardNN(7, 1, self.alpha, actor=False, HP=self.HP)
        self.memory = PPOMemory(batch_size)
        self.n_epochs = n_epochs

    def remember(self, state, action, probs, vals, reward, temp, done, current_episode):
        self.memory.store_memory(
            state, action, probs, vals, reward, temp, done, current_episode
        )

    def remember_location(self, location, idx):
        self.memory.store_location(location, idx)

    def save_models(self, path):
        # Save the actor1 and actor2 models
        self.actor1.save_network(path, suffix="actor1")
        self.actor2.save_network(path, suffix="actor2")

        # Save the critic model
        self.critic.save_network(path, suffix="critic")

        # Save the combined optimizer state for actor1 and actor2
        combined_optimizer_path = os.path.join(
            os.path.dirname(path), "combined_optimizer_" + os.path.basename(path)
        )
        torch.save(
            {"optimizer_state_dict": self.combined_optimizer.state_dict()},
            combined_optimizer_path,
        )

    def forward_actor1(self, observation, locations, temp):
        # Prepare batch data
        batch_data = []
        for i in range(len(observation)):
            batch_data.append(
                [
                    locations[observation[i - 1]][0],
                    locations[observation[i - 1]][1],
                    locations[observation[i]][0],
                    locations[observation[i]][1],
                    locations[observation[(i + 1) % (len(observation))]][0],
                    locations[observation[(i + 1) % (len(observation))]][1],
                    temp,
                ]
            )

        # Convert to PyTorch tensor
        batch_data = torch.tensor(batch_data, dtype=torch.float32).to(
            self.actor1.device
        )  # Shape: (NB_POINT, 6)

        # Pass the batch through actor1
        logits = self.actor1(batch_data)  # Ensure actor1 supports batch processing
        logits = logits.view(-1)

        # Apply Softmax to get probability distribution
        probs = torch.softmax(logits, dim=0)  # Normalize over the batch dimension

        return probs

    def forward_actor2(self, observation, locations, idx_first_node, temp):
        # Prepare batch data
        batch_data = []
        for i in range(len(observation)):  # Avoid out-of-bounds indexing
            if (
                i == idx_first_node
                or i == (idx_first_node + 1) % len(observation)
                or i == idx_first_node - 1
            ):
                continue
            batch_data.append(
                [
                    locations[observation[idx_first_node - 1]][0],
                    locations[observation[idx_first_node - 1]][1],
                    locations[observation[idx_first_node]][0],
                    locations[observation[idx_first_node]][1],
                    locations[observation[(idx_first_node + 1) % len(observation)]][0],
                    locations[observation[(idx_first_node + 1) % len(observation)]][1],
                    locations[observation[i - 1]][0],
                    locations[observation[i - 1]][1],
                    locations[observation[i]][0],
                    locations[observation[i]][1],
                    locations[observation[(i + 1) % (len(observation))]][0],
                    locations[observation[(i + 1) % (len(observation))]][1],
                    temp,
                ]
            )

        # Convert to PyTorch tensor
        batch_data = torch.tensor(batch_data, dtype=torch.float32).to(
            self.actor2.device
        )  # Shape: (NB_POINT, 12)

        # Pass the batch through actor2
        logits = self.actor2(batch_data)  # Ensure actor2 supports batch processing
        logits = logits.view(-1)

        # Apply Softmax to get probability distribution
        probs = torch.softmax(logits, dim=0)  # Normalize over the batch dimension

        return probs

    def forward_critic(self, locations, observation, temp):
        # Prepare batch data
        batch_data = []
        for i in range(len(observation)):  # Avoid out-of-bounds indexing
            batch_data.append(
                [
                    locations[observation[i - 1]][0],
                    locations[observation[i - 1]][1],
                    locations[observation[i]][0],
                    locations[observation[i]][1],
                    locations[observation[(i + 1) % (len(observation))]][0],
                    locations[observation[(i + 1) % (len(observation))]][1],
                    temp,
                ]
            )

        # Convert to PyTorch tensor
        batch_data = torch.tensor(batch_data, dtype=torch.float32).to(
            self.critic.device
        )  # Shape: (batch_size, 6)

        # Pass the batch data through the critic network
        value = self.critic(batch_data).sum()  # Ensure critic can handle batch inputs

        return value

    def choose_action(self, observation, episode, temp):
        locations = self.memory.get_locations(episode)

        # Get probabilities from actor1
        probs1 = self.forward_actor1(observation, locations, temp)

        # Sample first node based on probabilities
        first_node_idx = torch.multinomial(
            probs1, num_samples=1
        ).item()  # Sample from probs1
        first_node = observation[first_node_idx]

        # Get probabilities from actor2 using first_node and its neighbors
        probs2 = self.forward_actor2(observation, locations, first_node_idx, temp)

        # Sample second node based on probabilities
        second_node_idx = torch.multinomial(
            probs2, num_samples=1
        ).item()  # Sample from probs2

        # Log probabilities for PPO
        log_probs1 = torch.log(probs1[first_node_idx] + 1e-10)  # Avoid log(0)
        log_probs2 = torch.log(probs2[second_node_idx] + 1e-10)

        if first_node_idx == 0:
            second_node_idx += 2
        elif first_node_idx == len(observation) - 1:
            second_node_idx -= 1
        elif second_node_idx >= first_node_idx - 1:
            second_node_idx += 3

        # Utilisez second_node_idx pour accéder au nœud correspondant
        second_node = observation[second_node_idx]

        # Compute value from critic
        value = self.forward_critic(locations, observation, temp)

        # probs = [log_probs1, log_probs2]
        probs = log_probs1 + log_probs2
        action = [first_node, second_node]
        value = torch.squeeze(value)

        return action, probs, value

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

        return torch.tensor(all_advantages, dtype=torch.float)

    def calculate_probability_ratio(self, old_policy_probs, states, episode, temp):
        states = states.tolist()
        new_probs = []
        critic_value = []
        for s, e, t in zip(states, episode, temp):
            _, new_prob, new_value = self.choose_action(
                s, e, t
            )  # new_prob ->[prob_action1, prob_action2], new_value -> value critic
            new_probs.append(new_prob)
            critic_value.append(new_value)
        new_probs = torch.stack(new_probs).to(self.actor1.device)
        critic_value = torch.stack(critic_value).to(self.actor1.device)
        # r_phi = new_probs[:, 0] - old_policy_probs[:, 0]
        # r_theta = new_probs[:, 1] - old_policy_probs[:, 1]
        # prob_ratio = torch.exp(r_phi * r_theta)  # prob_ratio -> (len(states),)
        prob_ratio = torch.exp(new_probs - old_policy_probs)
        return prob_ratio, critic_value

    def update_lr(self):
        self.scheduler.step()
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
                vals_arr.clone().detach().to(torch.float32).to(self.actor1.device)
            )  # Convert values to tensor in float32

            old_prob_arr = (
                old_prob_arr.clone().detach().to(torch.float32).to(self.actor1.device)
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
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantages[batch]
                )
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Ctiric loss
                returns = advantages[batch] + vals_arr[batch]  # R_t = A_t + V(s)
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                # Backpropagation and optimization step
                self.combined_optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                # Apply gradient clipping (e.g., max norm of 0.5)
                torch.nn.utils.clip_grad_norm_(self.actor1.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.actor2.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.combined_optimizer.step()
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
