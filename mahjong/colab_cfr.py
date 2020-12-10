import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import NFSPAgent
from rlcard.agents import CFRAgent
from rlcard.agents import DeepCFR
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

env = rlcard.make('mahjong', config={'seed': 0, 'allow_step_back': True})
eval_env = rlcard.make('mahjong', config={'seed': 0})
path_prefix = '/content/drive/MyDrive/5011/'
# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 100
evaluate_num = 1000
episode_num = 10000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 64

# The paths for saving the logs and learning curves
log_dir = f'{path_prefix}/experiments/mahjong_cfr_result/'

# Set a global seed
set_global_seed(0)

sess = tf.Session()

# Initialize a global step
global_step = tf.Variable(0, name='global_step', trainable=False)
tf.variable_scope(name_or_scope='global_step', reuse=tf.AUTO_REUSE)

# Set up the agents
agents = []
for i in range(4):
    agent = CFRAgent(env)
    agents.append(agent)

random_agent = RandomAgent(action_num=eval_env.action_num)
env.set_agents([agents[0], agents[1], agents[2], agents[3]])
eval_env.set_agents([agents[0], random_agent, random_agent, random_agent])

sess.run(tf.global_variables_initializer())

# Init a Logger to plot the learning curvefrom rlcard.agents.random_agent import RandomAgent

logger = Logger(log_dir)

for episode in range(episode_num):
    for agent in agents:
        agent.train()
    print('\rIteration {}'.format(episode), end='')
    # if episode % evaluate_every == 0:
    #     agent.save()  # Save model
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('CFR')

# Save model
save_dir = f'{path_prefix}models/mahjong_cfr'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saver = tf.train.Saver()
saver.save(sess, os.path.join(save_dir, 'model'))
