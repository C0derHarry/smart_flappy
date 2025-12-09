import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = "mps" if torch.backends.mps.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.batch_size = hyperparameters['batch_size']
        self.epsilon_start = hyperparameters['eppsilon_start']
        self.epsilon_end = hyperparameters['epsilon_end']
        self.epsilon_decay = hyperparameters['epsilon_decay']


    def run(self, is_training=True, render=False):

        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        input_parameters = env.observation_space.shape[0]
        output_parameters = env.action_space.n

        rewards_per_episode = []

        policy_net = DQN(input_parameters, output_parameters).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_start

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    # Exploration: random action
                    action = env.action_space.sample()
                else:
                    # Exploitation: action from policy network
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_values = policy_net(state_tensor)
                        action = q_values.argmax().item()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)

                # Accumulating reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                rewards_per_episode.append(episode_reward)

                if is_training:
                    # Storing experience in replay memory
                    memory.add((state, action, new_state, info.get('next_observation'), terminated))

                # Updating state
                state = new_state

if __name__ == "__main__":
    agent = Agent(hyperparameter_set='cartPole1')
    agent.run(is_training=True, render=True)