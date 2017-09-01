import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gym, time, numpy as np, tensorflow as tf, threading, tensorflow.contrib.slim as slim, config
from gym import wrappers
from functools import partial
from functions import *
from agent import *
from worker import *


state_shape = (config.FRAME_NUM, *config.IMAGE_SIZE)
process_frame_fn = partial(process_frame, size=config.IMAGE_SIZE)
send_mail_fn = partial(send_mail, fromaddr=config.REPORT_MAIL_FROM_ADDRESS, username=config.REPORT_MAIL_FROM_USERNAME, password=config.REPORT_MAIL_FROM_PASSWORD, smtp=config.REPORT_MAIL_SMTP, to=config.REPORT_MAIL_TO, subject=config.REPORT_MAIL_SUBJECT)
all_rewards = []
episodes = [0 for _ in range(config.WORKER_NUM)]
global_agent_lock = threading.Lock()
summary_lock = threading.Lock()
envs = [gym.make(config.GAME) for _ in range(config.WORKER_NUM)]
valid_actions = range(envs[0].action_space.n)

tf.reset_default_graph()

#envs[0] = wrappers.Monitor(envs[0], '/tmp/{}'.format(config.GAME), force=True)

#with tf.Session( config=tf.ConfigProto(device_count = {'GPU': 0}) ) as sess:
with tf.Session() as sess:

    global_agent = Agent(sess, 'global', state_shape, valid_actions, config.TRAINER)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global'))

    workers = []
    for i in range(config.WORKER_NUM):
        ID = i + 1
        name = 'worker{}'.format(ID)
        agent = Agent(sess, name, state_shape, valid_actions, config.TRAINER)
        workers.append( Worker(ID, envs[i], agent, global_agent, config.FRAME_NUM, config.BATCH_SIZE, process_frame_fn, global_agent_lock, summary_lock, all_rewards, episodes) )
        
    sess.run( tf.global_variables_initializer() )
    if config.LOAD_WEIGHTS:
        saver.restore(sess, config.SAVE_FILE)

    threads = []
    for worker in workers:
        thread_stop_event = threading.Event()
        t = threading.Thread(target=worker.work, args=(thread_stop_event,))
        t.start()
        threads.append( {'thread': t, 'stop_event': thread_stop_event} )
        time.sleep(0.5)

    reward_num = 0
    try:
        while True:
            time.sleep(config.REPORT_INTERVAL)
            #save_file = 'save/a3c_save_{}.ckpt'.format(int(time.time()))
            with global_agent_lock:
                saver.save(sess, config.SAVE_FILE)
                #saver.save(sess, save_file)

            with summary_lock:
                max_reward = np.amax(all_rewards[reward_num:])
                avg_reward = round( sum(all_rewards[reward_num:])/len(all_rewards[reward_num:]), 1 )
                global_episode = sum(episodes)
                reward_num = len(all_rewards)

            summary = 'Global episode: {} | Average reward: {} | Max reward: {}'.format(global_episode, avg_reward, max_reward)
            print('\n'+summary+'\n')

            if config.SEND_MAIL:
                send_mail_fn(message=summary)

    except KeyboardInterrupt:
        raise
    except:
        for thread in threads:
            thread['stop_event'].set()
        with global_agent_lock:
            saver.save(sess, config.BACKUP_FILE)
        if config.SEND_MAIL:
            send_mail_fn(message='End')
