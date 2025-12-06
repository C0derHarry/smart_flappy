import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

class Agent:
    def run(self, is_training=True, render=False):

        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)

        input_parameters = env.observation_space.shape[0]
        output_parameters = env.action_space.n

        policy_net = DQN(input_parameters, output_parameters).to(device)

        obs, _ = env.reset()
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            obs, reward, terminated, _, info = env.step(action)
            
            # Checking if the player is still alive
            if terminated:
                break

        env.close()