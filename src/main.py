from env import ENV
from agent2 import Agent
from HP import _HP, get_script_arguments
from Logger import WandbLogger
from tqdm import tqdm
import numpy as np
from plot import plot_tsp_solutions
import copy as cp
from collections import deque

HP = _HP("src/HP.yaml")
HP.update(get_script_arguments(HP.keys()))

if HP["LOG"]:
    WandbLogger.init(None, 3, HP)

if __name__ == "__main__":
    env = ENV(
        HP["NB_POINT"],
        # HP["EPISODE_LEN"],
        HP["OR_TOOLS_TIME"],
        HP["TEMP_FINAL"],
        HP["K"],
    )
    N = HP["TRAIN_EVERY_N_EPISODES"]  # Number of steps before updating the policy
    batch_size = HP["BATCH_SIZE"]  # Size of mini-batches for PPO updates
    n_epochs = HP["EPOCHS_TRAIN"]  # Number of epochs to train on the collected data
    alpha = HP["ALPHA"]  # Learning rate for both actor and critic
    gamma = HP["GAMMA"]  # Discount factor
    gae_lambda = HP["GAE_LAMBDA"]  # Generalized Advantage Estimation lambda
    policy_clip = HP["POLICY_CLIP"]  # Clipping parameter for the PPO policy loss
    agent = Agent(
        HP=HP,
        batch_size=batch_size,
        n_epochs=n_epochs,
        alpha=alpha,
        gamma=gamma,
        gae_lambda=gamma,
        policy_clip=gamma,
    )

    n_games = HP["TOTAL_EPISODES"]  # Total number of episodes to train

    # Initialize variables to track performance
    mean_final_reward = deque(maxlen=50)
    mean_reward_episode = deque(maxlen=50)
    vs_ortools_list = deque(maxlen=50)
    best = deque(maxlen=50)
    nb_train = 0
    for current_episode in tqdm(range(n_games)):
        done = False
        score = []  # Accumulated reward for the current episode
        # Update the policy and value networks every N steps
        if current_episode % N == 0 and current_episode != 0:
            tot_loss, actor_losses, critic_losses = agent.learn()
            if HP["LOG"]:
                for tot, ac_loss, cr_loss in zip(tot_loss, actor_losses, critic_losses):
                    nb_train += 1
                    logs = {
                        "total_loss": tot,
                        "actor_loss": ac_loss,
                        "critic_loss": cr_loss,
                        "loss_step": nb_train,
                    }
                    WandbLogger.log(logs)
        observation, locations, temp, SA_dist = env.reset()
        agent.remember_location(locations, env.distance_matrix, current_episode)
        while not done:
            action, prob, val = agent.choose_action(observation, current_episode, temp)
            observation_, reward, temp_, done, info = env.step(action)
            score.append(reward)
            agent.remember(
                observation,
                action,
                prob.item(),
                val.item(),
                reward,
                temp,
                done,
                current_episode,
            )
            observation = observation_
            temp = temp_
        if current_episode % 100 == 0 and HP["PLOT"]:
            tmp = cp.deepcopy(env.best_solution)
            tmp.append(tmp[0])
            tmp2 = cp.deepcopy(env.initial_solution)
            tmp2.append(tmp2[0])
            plot_tsp_solutions(
                locations,
                env.solution,
                tmp,
                tmp2,
                "OR_TOOLS",
                "PPO",
                "Initial",
                env.solution_distance,
                env.best_distance,
                env.initial_distance,
                None,
            )

        vs_ortools = env.best_distance - env.solution_distance
        vs_ortools_list.append(vs_ortools)
        final_reward = env.final
        mean_final_reward.append(final_reward)
        mean_reward_episode.append(np.mean(score))
        best.append(env.best_distance)
        logs = {
            "SA Distance": SA_dist,  # Distance found by the SA algorithm
            "Best Distance": env.best_distance,  # Best distance found so far
            "Average_best_distance_50_ep": np.mean(best),  # Average best distance
            "Performance": final_reward,  # Difference between the initial and
            #  best distance (should go up)
            "Avreage_performance_50_ep": np.mean(
                mean_final_reward
            ),  # Average difference between the initial and
            #  best distance (should go up)
            "Avreage_score_per_episodes": np.mean(
                score
            ),  # Average reward per episode (should go up)
            "Avreage_score_50_ep": np.mean(
                mean_reward_episode
            ),  # Average reward over the last 10 episodes (should go up)
            "VS_ortools": vs_ortools,  # Value of the solution compared to OR-Tools
            # (should go down)
            "Avreage_VS_ortools_50_ep": np.mean(
                vs_ortools_list
            ),  # Average VS OR-Tools the last 10 episodes (should go down)
            "episode": current_episode,
        }
        if HP["LOG"]:
            WandbLogger.log(logs)
        if current_episode == 0:
            best_score = final_reward
        if final_reward > best_score:
            best_score = final_reward
            WandbLogger.log_model(
                agent.save_models, logs["Avreage_VS_ortools_50_ep"], current_episode
            )

    if HP["LOG"]:
        WandbLogger.close()
    print("... training complete ...")
