#!./venv/bin/python 
import sys, os, subprocess, shutil
#sys.path.append('/media/aakash/active/_xosuit/py_utils/')
#from autodep import install # local_module
#install(script_path=__file__, python_version='3.10.12', sys_reqs=['cuda==11.8'])

from headers import *
import config

# Load the shared library
lidar_lib = ctypes.CDLL('./lidar_readings.so')

# Define the argument and return types for the C++ function
lidar_lib.lidar_readings.argtypes = (
    ctypes.c_int,  # x (robot's x position)
    ctypes.c_int,  # y (robot's y position)
    ctypes.c_int,  # grid_size (size of the grid)
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # grid (2D grid array)
    ctypes.POINTER(ctypes.c_int)  # readings (output array for LiDAR distances)
)

per_episode_stats = {'avg_reward_per_step':[], 'total_loss':[], 'heatmaps':[], 'q_values':[], 'visited':[]}
per_episode_collisions = {'explored':[], 'collision_with_wall':[], 'collision_with_obstacle':[], 'revisited':[]}

def moving_average(numbers, N):
    """
    Computes the moving average of a list of numbers over a window of size N.
    
    Parameters:
    numbers (list): The input list of numbers.
    N (int): The size of the moving average window.
    
    Returns:
    list: The moving average of the input list.
    """
    cumsum = np.cumsum(np.insert(numbers, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def fill_surrounded_zeros(arr):
    '''
    This function iteratively fills the occupancy grid wherever the grid element is surrounded by more than 2 obstacles
    '''
    # Define a 3x3 kernel that will be used to count neighboring 1s
    kernel = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])
    while True:
        # Convolve the kernel with the array to count surrounding 1s for each element
        neighbor_count = convolve(arr, kernel, mode='constant', cval=0)
        
        # Find all positions where the value is 0 and the number of surrounding 1s > 2
        surrounded_positions = (arr == 0) & (neighbor_count > 2)

        # If no positions, then no changes to be made. Break out
        if not np.any(surrounded_positions):
            break
        #else:
        #    tqdm.write(str(np.any(surrounded_positions)))
        
        # Set these positions to 1
        arr[surrounded_positions] = 1
    
    return arr

# Environment
class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=50, obstacle_prob=config.obstacle_probability):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.obstacle_prob = obstacle_prob
        
        # Action space: 4 possible actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: lidar sensor reading + robot's position (flattened)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size * 2,), dtype=np.float32)

        # Variables for efficiency
        self._ten_cell_range = range(1, 11)
        self.action_effects = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right
        self.directions = self.action_effects    # for lidar : up, down, left, right
        self._empty_lidar_readings = [None, None, None, None] #deque(maxlen=10000)
        self.heatmap = None
        self.reset()

    def reset(self):
        # Reset environment to initial state
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Place obstacles randomly
        _ = list(range(self.grid_size))[1:-1] # dont place obstacles right next to wall, it makes closed spaces 
        for i in _:
            for j in _:
                if np.random.random() < self.obstacle_prob:
                    self.grid[i, j] = 1  # Obstacle
        
        # Simplify Obstacles
        self.grid = fill_surrounded_zeros(self.grid)

        # Initialize robot at random free location
        _c = 0
        while True:
            #print('Setting robot pos : ', _c)
            _c += 1
            if _c > self.grid.size:
                print('Cannot place robot. Possibly entire board is filled with obstacles.')
                print(str(self.grid))
                exit()
            self.robot_pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            if self.grid[self.robot_pos[0], self.robot_pos[1]] == 0: # If location is free / without obstacle
                break
        
        self.visited = np.zeros_like(self.grid) - self.grid # To keep track of explored areas
        self.heatmap =  copy.deepcopy(self.grid).astype(np.int64)*(-10) # np.zeros_like(self.grid).astype(np.int64)
        return self._get_obs()

    def _get_obs(self):
        # Return observation (position + lidar readings)
        lidar = self._lidar_readings_v2()
        return lidar #np.concatenate((lidar, self.robot_pos)) # Cannot use robot_pos since the agent should use only lidar readings
    '''
    def _lidar_readings(self):
        # Simulate a LiDAR sensor (4 directions) with max range of 10 cells

        # Convert Python 2D list (grid) to a ctypes 2D array (array of pointers)
        grid_ctypes = (ctypes.POINTER(ctypes.c_int) * self.grid_size)()
        for i in range(self.grid_size):
            grid_ctypes[i] = (ctypes.c_int * self.grid_size)(*self.grid[i])
        
        # Prepare the output array for the readings
        readings_ctypes = (ctypes.c_int * 4)()

        # Call the C++ function
        lidar_lib.lidar_readings(ctypes.c_int(self.robot_pos[0]), ctypes.c_int(self.robot_pos[1]), ctypes.c_int(self.grid_size), grid_ctypes, readings_ctypes)

        # Convert the readings back to a Python list
        readings = [readings_ctypes[i]/(2*self.grid_size) for i in range(4)] # TODO : Lidar readings shouldnt be more than 1 after normalisation, but they currently are.
        assert np.max(readings) <= 1, f'LiDAR readings too large : {readings}'
        return np.array(readings)
    '''
    def find_neighbors(self, arr, num):
        idx = np.searchsorted(arr, num) # Get insertion index
        smaller = arr[idx - 1] if idx > 0 else np.array(-1)  # Element just smaller
        greater = arr[idx] if idx < len(arr) else np.array(self.grid_size)    # Element just greater
        #print(type(smaller), type(greater))
        #print(smaller.shape, greater.shape)
        #print(smaller, greater)
        return np.array([smaller, greater])

    def _lidar_readings_v2(self):
        readings = copy.deepcopy(self._empty_lidar_readings)
        x_obstacles, y_obstacles = np.where(self.grid[self.robot_pos[0],:] == 1)[0], np.where(self.grid[:,self.robot_pos[1]] == 1)[0]
        #x_obstacles, y_obstacles = np.array(x_obstacles), np.array(y_obstacles)
        
        readings[0], readings[1] =  np.abs(self.find_neighbors(x_obstacles, self.robot_pos[1]) - self.robot_pos[1]) #-x_obstacles[x_obstacles < 0][-1], x_obstacles[x_obstacles > 0][0] # left, right
        readings[2], readings[3] =  np.abs(self.find_neighbors(y_obstacles, self.robot_pos[0]) - self.robot_pos[0]) #-y_obstacles[y_obstacles < 0][-1], y_obstacles[y_obstacles > 0][0] # up, down
        return readings

    def increase_obstacle_prob(self):
        if self.obstacle_prob < config.max_obstacle_prob:    
            self.obstacle_prob += config.obstacle_probability_increment

    def step(self, action):
        global p_bar
        # Define action effects
        
        new_pos = self.robot_pos + self.action_effects[action] #[self.robot_pos[0] + self.action_effects[action][0], self.robot_pos[1] + self.action_effects[action][1]]

        reward = config.time_step_penalty  # Time step penalty
        
        #explored_percetange = None
        status = {'explored':0, 'collision_with_wall':0, 'collision_with_obstacle':0, 'revisited':0}

        # Check for collision with walls or obstacles
        if ((0 <= new_pos) & (new_pos < self.grid_size)).all(): #0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            if not self.grid[tuple(new_pos)]:  # Free cell
                self.robot_pos = new_pos
                if not self.visited[tuple(self.robot_pos)]:  # Unvisited cell
                    reward += config.exploration_reward  # Reward for exploration
                    status['explored'] = 1
                    p_bar.n += 1 / self.visited.size
                    p_bar.refresh()
                else:
                    reward += config.revisiting_penalty  # Penalty for revisiting
                    status['revisited'] = 1
                self.visited[tuple(self.robot_pos)] = 1  # Mark cell as visited
                
                # Update heatmap
                self.heatmap[tuple(self.robot_pos)] += 1
            else:
                reward += config.object_collision_penalty  # Collision with obstacle
                status['collision_with_obstacle'] = 1
        else:
            reward += config.object_collision_penalty  # Collision with wall
            status['collision_with_wall'] = 1
        done = np.all(self.visited != 0)  # Task complete when all cells are visited

        return self._get_obs(), reward, done, {}, np.sum(self.visited) / self.visited.size, status
    
    def get_heatmap(self):
        return self.heatmap
    
    def get_visited(self):
        return self.visited

# DQN Agent
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Input size is 6
        self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim, device, lr=0.001, gamma=0.99, epsilon_start=config.epsilon_start, epsilon_end=config.epsilon_end, epsilon_decay=config.epsilon_decay):
        self.model = DQN(input_dim, output_dim).to(device)
        self.target_model = DQN(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Random action (exploration)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.model(state).argmax().item()  # Greedy action (exploitation)
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0  # Return 0 for both loss and q_value if there's insufficient memory
        
        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Get Q-values for current states and actions
        q_values = self.model(states).gather(1, actions)
        
        # Get max Q-values for the next states
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        
        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calculate loss (Mean Squared Error)
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Detach Q-values to return them without affecting the gradient
        q_value_mean = q_values.mean().item()  # You can log the mean Q-value
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return both the loss and Q-value
        return loss.item(), q_value_mean


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

p_bar = tqdm(range(config.num_episodes))
p_bar.n = 0

def add_dicts(dict1, dict2):
    return {_:dict1[_] + dict2[_] for _ in dict1.keys()}
    
def train_dqn(agent, env, num_episodes=config.num_episodes):
    global pbar, per_episode_stats, per_episode_collisions
    rewards_per_episode = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        step_counter = 0
        residual_patience = config.patience
        prev_explored_percentage = 0
        running_loss = 0
        running_q = 0
        status_counter = {'explored':0, 'collision_with_wall':0, 'collision_with_obstacle':0, 'revisited':0}
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, explored_percentage, status = env.step(action)
            status_counter = add_dicts(status_counter, status) #dict(Counter(status_counter) + Counter(status)) # Update status_counter
            agent.store_experience(state, action, reward, next_state, done)
            _loss, _q = agent.train()
            running_loss += _loss
            running_q += _q
            state = next_state
            total_reward += reward
            step_counter += 1
            if explored_percentage != prev_explored_percentage:    
                residual_patience = config.patience
                prev_explored_percentage = explored_percentage
            else:
                residual_patience -= 1
                if residual_patience <= 0:
                    p_bar.n = int(p_bar.n)
                    p_bar.update()
                    break
                    
            #_max_action_count -= 1
            #tqdm.write(f'Max action count : {_max_action_count}')
        
        #_max_action_count += config._max_action_count_increment
        rewards_per_episode.append(total_reward)
        per_episode_stats['avg_reward_per_step'].append(total_reward/step_counter)
        per_episode_stats['total_loss'].append(running_loss)
        per_episode_stats['heatmaps'].append(env.get_heatmap())
        per_episode_stats['q_values'].append(running_q)
        per_episode_stats['visited'].append(env.get_visited())
        per_episode_collisions = {_:(per_episode_collisions[_] + [status_counter[_]]) for _ in list(per_episode_collisions.keys())}

        agent.update_target_model()
        agent.decay_epsilon()
        env.increase_obstacle_prob()

        if episode % 10 == 0:
            tqdm.write(f"Episode {episode}, Total Reward: {total_reward}")
    p_bar.close()

    return rewards_per_episode

# Plot reward over episodes
def plot_rewards(rewards_per_episode):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def plot_everything(stats:list):
    # Example data
    
    y1, y1_name = moving_average(stats['avg_reward_per_step'], 7), 'Avg Reward per Step'
    y2, y2_name = moving_average(stats['total_loss'], 7), 'Total Loss'
    y3, y3_name = moving_average(stats['q_values'], 7), 'Q Values'
    x = range(len(y1))
    # Create the figure and the first axis
    fig, ax1 = plt.subplots()

    # Plot the first series
    line1, = ax1.plot(x, y1, 'g-', label=y1_name)
    ax1.set_ylabel(y1_name, color='g')
    ax1.yaxis.offsetText.set_visible(False)
    offset = ax1.yaxis.get_major_formatter().get_offset()
    ax1.yaxis.set_label_text(y1_name + " " + offset)

    # Create the second axis, sharing the same x-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(x, y2, 'b-', label=y2_name)
    ax2.set_ylabel(y2_name, color='b')
    ax2.yaxis.offsetText.set_visible(False)
    offset = ax2.yaxis.get_major_formatter().get_offset()
    ax2.yaxis.set_label_text(y2_name + " " + offset)

    # Create the third axis
    ax3 = ax1.twinx()

    # Offset the position of the third axis to the right
    ax3.spines['right'].set_position(('outward', 60))  # Shift by 60 points to the right
    line3, = ax3.plot(x, y3, 'r-', label=y3_name)
    ax3.set_ylabel(y3_name, color='r')
    ax3.yaxis.offsetText.set_visible(False)
    offset = ax3.yaxis.get_major_formatter().get_offset()
    ax3.yaxis.set_label_text(y3_name + " " + offset)

    # Create the legend in the top-left corner
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')  # Set the legend in the top-left corner
    plt.title('Convergence Plot')

    # Plot
    plt.savefig('convergence_plot.png', dpi=config.plot_dpi, bbox_inches='tight')  # 'bbox_inches' ensures no element is cut off

def plot_everything_else(stats:list):
    # {'explored':[], 'collision_with_wall':[], 'collision_with_obstacle':[], 'revisited':[]}
    
    y1, y1_name = moving_average(stats['collision_with_wall'],7), 'Collisions with Wall'
    y2, y2_name = moving_average(stats['collision_with_obstacle'],7), 'Collisions with Obstacles'
    y3, y3_name = moving_average(stats['revisited'],7), 'Re-visits'
    x = range(len(y1))

    # Create the figure and the first axis
    fig, ax1 = plt.subplots()

    # Plot the first series
    line1, = ax1.plot(x, y1, 'g-', label=y1_name)
    ax1.set_ylabel(y1_name, color='g')
    ax1.yaxis.offsetText.set_visible(False)
    offset = ax1.yaxis.get_major_formatter().get_offset()
    ax1.yaxis.set_label_text(y1_name + " " + offset)

    # Create the second axis, sharing the same x-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(x, y2, 'b-', label=y2_name)
    ax2.set_ylabel(y2_name, color='b')
    ax2.yaxis.offsetText.set_visible(False)
    offset = ax2.yaxis.get_major_formatter().get_offset()
    ax2.yaxis.set_label_text(y2_name + " " + offset)

    # Create the third axis
    ax3 = ax1.twinx()

    # Offset the position of the third axis to the right
    ax3.spines['right'].set_position(('outward', 60))  # Shift by 60 points to the right
    line3, = ax3.plot(x, y3, 'r-', label=y3_name)
    ax3.set_ylabel(y3_name, color='r')
    ax3.yaxis.offsetText.set_visible(False)
    offset = ax3.yaxis.get_major_formatter().get_offset()
    ax3.yaxis.set_label_text(y3_name + " " + offset)

    # Create the legend in the top-left corner
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')  # Set the legend in the top-left corner
    plt.title('Collisions & Revisits Per Episode')

    # Plot
    plt.savefig('collisions_and_revisits_per_epsiode.png', dpi=config.plot_dpi, bbox_inches='tight')  # 'bbox_inches' ensures no element is cut off

'''
def plot_heatmaps(stats:list):

    # List of 2D square numpy arrays
    array_list = stats['heatmaps']

    # Find the global minimum and maximum across all arrays for consistent color scaling
    vmin = np.min([arr.min() for arr in array_list])
    vmax = np.max([arr.max() for arr in array_list])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the first heatmap to initialize the colorbar
    sns.heatmap(array_list[0], ax=ax, cbar=True, square=True, cmap="viridis", vmin=vmin, vmax=vmax)

    # Function to update the heatmap in each frame
    def animate(i):
        ax.clear()  # Clear the axis to prevent overlap of successive heatmaps
        sns.heatmap(array_list[i], ax=ax, cbar=False, square=True, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Heatmap at Episode {i}")

    # Creating the animation
    ani = FuncAnimation(fig, animate, frames=len(array_list), interval=500, repeat=False)

    # Saving the animation as a GIF
    ani.save("heatmap_progression_with_colorbar.gif", writer=PillowWriter(fps=2))

    #plt.show()
'''

def plot_heatmaps(stats: list):
    # List of 2D square numpy arrays
    array_list = stats['heatmaps']

    # Find the global minimum and maximum across all arrays for consistent color scaling
    vmin = np.min([arr.min() for arr in array_list])
    vmax = np.max([arr.max() for arr in array_list])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the first heatmap to initialize the colorbar
    sns.heatmap(array_list[0], ax=ax, cbar=True, square=True, cmap="viridis", vmin=vmin, vmax=vmax)

    # Function to update the heatmap in each frame
    def animate(i):
        ax.clear()  # Clear the axis to prevent overlap of successive heatmaps
        sns.heatmap(array_list[i], ax=ax, cbar=False, square=True, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Heatmap at Episode {i}")
        
        # Save the individual frame as an image
        plt.savefig(f'heatmap_frame_{i}.png', dpi=300, bbox_inches='tight')

    # Creating the animation
    ani = FuncAnimation(fig, animate, frames=len(array_list), interval=500, repeat=False)

    # Save the animation as a GIF
    ani.save("heatmap_progression_with_colorbar.gif", writer=PillowWriter(fps=2))

    # Optionally display the plot
    # plt.show()

def plot_visits(stats: list):
    # List of 2D square numpy arrays
    array_list = stats['visited']

    # Find the global minimum and maximum across all arrays for consistent color scaling
    #vmin = 0#np.min([arr.min() for arr in array_list])
    #vmax = 1#np.max([arr.max() for arr in array_list])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the first heatmap to initialize the colorbar
    sns.heatmap(array_list[0], ax=ax, cbar=True, square=True, cmap="viridis") #, vmin=vmin, vmax=vmax)

    # Function to update the heatmap in each frame
    def animate(i):
        ax.clear()  # Clear the axis to prevent overlap of successive heatmaps
        sns.heatmap(array_list[i], ax=ax, cbar=False, square=True, cmap="viridis") #, vmin=vmin, vmax=vmax)
        ax.set_title(f"Visited Grid at Episode {i}")
        
        # Save the individual frame as an image
        plt.savefig(f'visited_frame_{i}.png', dpi=300, bbox_inches='tight')

    # Creating the animation
    ani = FuncAnimation(fig, animate, frames=len(array_list), interval=500, repeat=False)

    # Save the animation as a GIF
    ani.save("visited_progression_with_colorbar.gif", writer=PillowWriter(fps=2))

    # Optionally display the plot
    # plt.show()

def plot_visit_grid(stats: list):
    # Example list of 2D square numpy arrays
    array_list = stats['visited']

    # Find global min and max for the color scale
    min_val = min([arr.min() for arr in array_list])
    max_val = max([arr.max() for arr in array_list])

    # Parameters for the grid
    m = config.grid_columns  # Number of columns in the grid
    n = len(array_list) // m + (len(array_list) % m > 0)  # Number of rows

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), squeeze=False)

    # Plot each heatmap in the grid
    heatmaps = []
    for i, ax in enumerate(axes.flat):
        if i < len(array_list):
            # Plot heatmap but suppress color bar for individual heatmaps
            heatmap = sns.heatmap(array_list[i], ax=ax, cbar=False, vmin=min_val, vmax=max_val, cmap="viridis", square=True)
            heatmaps.append(heatmap)
            ax.set_title(f"Episode {i}")
        else:
            ax.axis('off')  # Turn off unused subplots

def plot_heatmap_grid(stats: list):
    # Example list of 2D square numpy arrays
    array_list = stats['heatmaps']

    # Find global min and max for the color scale
    min_val = min([arr.min() for arr in array_list])
    max_val = max([arr.max() for arr in array_list])

    # Parameters for the grid
    m = config.grid_columns  # Number of columns in the grid
    n = len(array_list) // m + (len(array_list) % m > 0)  # Number of rows

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), squeeze=False)

    # Plot each heatmap in the grid
    heatmaps = []
    for i, ax in enumerate(axes.flat):
        if i < len(array_list):
            # Plot heatmap but suppress color bar for individual heatmaps
            heatmap = sns.heatmap(array_list[i], ax=ax, cbar=False, vmin=min_val, vmax=max_val, cmap="viridis", square=True)
            heatmaps.append(heatmap)
            ax.set_title(f"Episode {i}")
        else:
            ax.axis('off')  # Turn off unused subplots

    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on the right for the color bar

    # Add a common color bar (legend) for all subplots
    # To link the color bar to all subplots, we need to use a collection from any of the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the color bar on the right
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])  # No data to pass here, we use the norm already defined
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_ylabel("Color Intensity")  # Set the label for the color bar

    plt.suptitle('Episode Heatmaps', y=1.02)

    # Save the figure
    plt.savefig("heatmap_grid_with_legend.png", dpi=config.plot_dpi, bbox_inches='tight')
    #plt.show()

# Initialize environment and agent
env = GridWorldEnv(grid_size=config.grid_size) # 50
input_dim = 4  # LiDAR readings (4) 
output_dim = env.action_space.n
device = config.device
tqdm.write(f'Device = {device}')
agent = DQNAgent(input_dim, output_dim, device)

# Train the agent
rewards_per_episode = train_dqn(agent, env, num_episodes=config.num_episodes)

# Plot the rewards
#plot_rewards(rewards_per_episode)

plot_everything(stats=per_episode_stats)
print('Convergence plot saved.')

plot_everything_else(stats=per_episode_collisions)
print('Collisions and revisits plot saved.')
#plot_heatmap_grid(stats=per_episode_stats)
#print('Heatmaps grid saved.')

plot_heatmaps(stats=per_episode_stats)
print('Heatmaps saved.')

plot_visits(stats=per_episode_stats)
print('Visited plots saved. Exiting.')