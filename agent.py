import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random
from torch import nn

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
        self.network_update_frequency = hyperparameters['network_update_frequency']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None


    def run(self, is_training=True, render=False):

        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        #observation space and possible actions
        input_parameters = env.observation_space.shape[0]
        output_parameters = env.action_space.n

        rewards_per_episode = []

        policy_net = DQN(input_parameters, output_parameters).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_start

            target_DQN = DQN(input_parameters, output_parameters).to(device)
            target_DQN.load_state_dict(policy_net.state_dict())

            #track number of steps taken. Used for syncing policy => target network
            step_count = 0

            self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate)

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
                    step_count += 1

                # Updating state
                state = new_state

                if len(self) > self.batch_size:
                    # Sample a batch of experiences from replay memory
                    mini_batch = memory.sample(self.batch_size)
                    # Here you would typically process the experiences and update the policy network
                    self.optimize(policy_net, target_DQN, mini_batch)

                    if step_count > self.network_update_frequency:
                        target_DQN.load_state_dict(policy_net.state_dict())
                        step_count = 0


    def optimize(self, policy_net, target_DQN, mini_batch):
        
        for state, action, next_state, reward, terminated in mini_batch:

            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target = reward + self.discount_factor * target_DQN(next_state).max()

            current_q_value = policy_net(state)

            #calculate loss
            loss = self.loss_fn(current_q_value, target)

            #optimize the model 
            self.optimizer.zero_grad()      #clear previous gradients
            loss.backward()                 #backpropagate the loss
            self.optimizer.step()           #update the model parameters i.e. weights and biases


if __name__ == "__main__":
    agent = Agent(hyperparameter_set='cartPole1')
    agent.run(is_training=True, render=True)