import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from frozen_lake import Shipping
from mapping import OceanMapper, Coordinate
# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        
        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, 256) 
        self.fc3 = nn.Linear(256, 256) 
        self.out = nn.Linear(256, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        # x = F.relu(self.fc8(x))
        # x = F.relu(self.fc9(x))
        # x = F.relu(self.fc10(x))
        # x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))# Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class FrozenLakeDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 256           # size of the training data set sampled from the replay memory
    
    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)
    def __init__(self):
        mapper = OceanMapper(grid_size=100)
        top_left = Coordinate(lat=25.0, lon=-85.0)  # Near Florida
        bottom_right = Coordinate(lat=19.0, lon=-75.0)  # Near Cuba
        desc1 = mapper.get_ocean_map(top_left, bottom_right)
        self.env = Shipping(
            time_of_day=0,
        fuel_model=None,
        speed_model=None,
        render_mode = 'human',
        desc=desc1,
        map_name=None,
        is_slippery=False
        )
    # Train the FrozeLake environment
    def train(self, episodes, render=False, is_slippery=False):
        # Create FrozenLake instance
        
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for i in range(episodes):
            state = self.env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    
            # print('nig')
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            step_count=0
            while(not terminated and not truncated):
                # print('ga')
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    values = np.array([0,1,2,3])
                    probabilities = np.array([0.15, 0.35, 0.35, 0.15])

# Ensure the probabilities sum to 1
                    probabilities /= probabilities.sum()

# Create a random generator
                    rng = np.random.default_rng()

# Sample from the custom distribution
                    sample = rng.choice(values, size=1, p=probabilities)
                    action =sample[0] # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,win = self.env.step(action,step_count)
                # print(terminated)
                self.env.render()
                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if step_count>=500 or win or terminated or truncated:
                rewards_per_episode[i] = reward
                
            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)!=0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        
                print('learn',i,epsilon,step_count)
                # Decay epsilon
                epsilon = epsilon*0.95
                epsilon_history.append(epsilon)
                print(np.count_nonzero(self.env.visit_dict >=0))
                
                # Copy policy network to target network after a certain number of steps
                if step_count%self.network_sync_rate==0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    

        # Close environment
        # env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "frozen_lake_dqll.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('frozen_lake_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery=False):
        # Create FrozenLake instance
         
        
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n


        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("frozen_lake_dqll.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        # self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = self.env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = self.env.step(action,0)
                self.env.render()
        self.env.close()
        
    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        dict1={}
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]
            try:
                dict1[best_action]+=1
            except:
                dict1[best_action]=1    
            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            # print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            # if (s+1)%4==0:
            #     print() # Print a newline every 4 states
            if s%200==0:
                print(dict1)
        print(dict1)    
    def validate(self):
        self.env.reset()
        for i in range(100):
            self.env.step(1,0)
            self.env.render()
        for i in range(100):
            self.env.step(2,0)
            self.env.render()    
if __name__ == '__main__':

    frozen_lake = FrozenLakeDQL()
    is_slippery = False
    # frozen_lake.validate()
    frozen_lake.train(50, is_slippery=is_slippery)
    frozen_lake.test(10, is_slippery=is_slippery)
