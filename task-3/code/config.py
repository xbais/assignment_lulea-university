##############
#   CONFIG   #
##############
 
grid_size = 50
device = 'cuda'
num_episodes = 50 #150 # 200
obstacle_probability = 0.15 #0.1

# Rewards
time_step_penalty = -1
object_collision_penalty = -5
exploration_reward = 5
revisiting_penalty = -2.5

# Patience
patience = 1000 # 2000 # 5000 # 10000 # If visited cells does not improve within these many steps, then move to next episode --> Need : to avoid getting stuck on unsolvable board situations

#_max_action_count_increment = 1000
max_obstacle_prob = 0.15
obstacle_probability_increment = (max_obstacle_prob-obstacle_probability)/(num_episodes/2)

# Plot
grid_columns = 4
plot_dpi = 300

# Epsilon : Exploration versus Exploitation
epsilon_start = 4.0
epsilon_end = 1 #0.5 # 0.01
epsilon_decay = 0.996 # Values closer to 1 will make the epsilon decay slower