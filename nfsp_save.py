''' An example of learning a NFSP Agent on Mahjong
'''

import shutil

import tensorflow as tf
import os
import time
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
env = rlcard.make('mahjong', config={'seed': 0})
eval_env = rlcard.make('mahjong', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 100
evaluate_num = 100
episode_num = 400

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 64

# The paths for saving the logs and learning curves
log_dir = f'./experiments/nfsp_result_{time.time()}/'
path_prefix = './'
save_dir_pre = f'{path_prefix}models/nfsp_'
save_dir_main = f'{save_dir_pre}main/'
save_dir_last = f'{save_dir_pre}last/'
# Set a global seed
set_global_seed(0)


def save_model(sess, saver):
    # Save model
    # save_dir = f'{save_dir_last}/model'
    save_dir2 = f'{save_dir_main}/model.ckpt'
    if os.path.exists(save_dir_last):
        shutil.rmtree(save_dir_last)
    # os.makedirs(save_dir)
    if os.path.exists(save_dir_main):
        shutil.copytree(save_dir_main, save_dir_last)
        shutil.rmtree(save_dir_main)
    os.makedirs(save_dir_main)
    # os.makedirs(save_dir2)
    saver.save(sess, save_dir2)


def load_sess(sess, saver):
    sl = os.path.exists(save_dir_last)
    sm = os.path.exists(save_dir_main)
    if not sl and not sm:
        pass
    elif sl and not sm:
        module_file = tf.train.latest_checkpoint(save_dir_last)
        saver.restore(sess, module_file)
    else:
        module_file = tf.train.latest_checkpoint(save_dir_main)
        saver.restore(sess, module_file)


with tf.Session() as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(sess,
                          scope='nfsp' + str(i),
                          action_num=env.action_num,
                          state_shape=env.state_shape,
                          hidden_layers_sizes=[512, 1024, 2048, 1024, 512],
                          anticipatory_param=0.5,
                          batch_size=256,
                          rl_learning_rate=0.00005,
                          sl_learning_rate=0.00001,
                          min_buffer_size_to_learn=memory_init_size,
                          q_replay_memory_size=int(1e5),
                          q_replay_memory_init_size=memory_init_size,
                          train_every=train_every,
                          q_train_every=train_every,
                          q_batch_size=256,
                          q_mlp_layers=[512, 1024, 2048, 1024, 512])
        agents.append(agent)
    random_agent = RandomAgent(action_num=eval_env.action_num)

    env.set_agents(agents)
    eval_env.set_agents([agents[0], random_agent, random_agent, random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    load_sess(sess, saver)
    # Init a Logger to plot the learning curvefrom rlcard.agents.random_agent import RandomAgent

    logger = Logger(log_dir)

    for episode in range(episode_num):

        # First sample a policy for the episode
        for agent in agents:
            agent.sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i in range(env.player_num):
            for ts in trajectories[i]:
                agents[i].feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])
            save_model(sess, saver)
            print('The episode is : ', episode)
    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('NFSP')

    save_model(sess, saver)
