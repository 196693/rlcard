''' An example of learning a Deep-Q Agent on Mahjong
'''
import shutil

import tensorflow as tf
import os
import time
import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
env = rlcard.make('mahjong', config={'seed': 0})
eval_env = rlcard.make('mahjong', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 100
evaluate_num = 1000
episode_num = 1000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = f'./experiments/new_dqn_4/mahjong_dqn_4_result_{time.time()}/'
path_prefix = './'
save_dir_pre = f'{path_prefix}models/mahjong_dqn4_'
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


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agents = []
    for i in range(4):
        agent = DQNAgent(sess,
                         scope=f'dqn_{i}',
                         action_num=env.action_num,
                         replay_memory_size=20000,
                         replay_memory_init_size=memory_init_size,
                         train_every=train_every,
                         state_shape=env.state_shape,
                         mlp_layers=[512, 512])
        agents.append(agent)

    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents(agents)
    eval_env.set_agents([agents[0], random_agent, random_agent, random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    load_sess(sess, saver)

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i in range(4):
            for ts in trajectories[i]:
                agent.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])
            save_model(sess, saver)
            logger.log(f'The episode is : {episode}')
    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('DQN')

    save_model(sess, saver)
