import torch
import torch.nn as nn
import torch.nn.functional as F 

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DQN, self).__init__()

        # Define the network layers -> only defining hidden and output layers, input layer is defined by input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    # Example usage
    input_dim = 12  # Example input dimension
    output_dim = 2  # Example output dimension (number of actions)
    model = DQN(input_dim, output_dim)
    state = torch.randn(1, input_dim)  # Example state
    q_values = model(state)
    print(q_values)