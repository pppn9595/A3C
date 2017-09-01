import numpy as np, tensorflow as tf, time
from functions import *

class Worker:
    def __init__(self, ID, env, agent, global_agent, frame_num, batch_size, process_frame_fn, global_agent_lock, summary_lock, summary_all_rewards, summary_episodes):
        self.ID = ID
        self.env = env
        self.agent = agent
        self.global_agent = global_agent
        self.frame_num= frame_num
        self.batch_size = batch_size
        self.process_frame_fn = process_frame_fn
        self.global_agent_lock = global_agent_lock
        self.summary_lock = summary_lock
        self.summary_all_rewards = summary_all_rewards
        self.summary_episodes = summary_episodes

    def train(self, episode_buffer, bootstrap = 0):
        states = []
        actions = []
        rewards = []
        values = []
        for state, action, reward, value in episode_buffer:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)

        states = self.agent.format_state(states)
        actions = np.array(actions)
        values = np.array(values)

        discounted_rewards = discount_rewards(rewards, bootstrap)

        advantages = discounted_rewards - values

        with self.global_agent_lock:
            self.global_agent.train(states, actions, discounted_rewards, advantages)
            self.agent.update_from_global()

    def run_episode(self):
        state = start_game(self.env, self.frame_num, self.process_frame_fn, True)
        done = False
        episode_reward = 0
        #lives = 5
        episode_buffer = []

        while not done:

            #self.env.render()

            action, value = self.agent.get_action_value(state)

            frame, reward, done, info = self.env.step(action)
            next_state = np.append(state[1:, :, :], self.process_frame_fn(frame), axis=0) # add new frame to the actual state

            episode_reward += reward

            #if info['ale.lives'] != lives or done:
            #    reward -= 1

            episode_buffer.append( [state[:], action, reward, value] ) # store train data

            if len(episode_buffer) == self.batch_size and not done:  # if memory is full train the model
                '''if reward != -1:
                    bootstrap = self.agent.get_action_value(next_state)[1]
                else:
                    bootstrap = 0'''
                bootstrap = self.agent.get_action_value(next_state)[1] # predict the future reward

                self.train(episode_buffer, bootstrap)
                episode_buffer = []

            #if info['ale.lives'] != lives and not done:
            #    next_state = start_game(self.env, self.frame_num, self.process_frame_fn)
            #    lives = info['ale.lives']

            state = next_state

            time.sleep(0.001) # with this line we could use more thread than CPU's thread number

        self.train(episode_buffer, 0) # future reward is 0, because the episode is over

        return episode_reward

    def work(self, stop_event):
        episode = 1

        with self.global_agent_lock:
            self.agent.update_from_global()
        while not stop_event.is_set():
            reward = self.run_episode()
            with self.summary_lock:
                self.summary_episodes[self.ID - 1] = episode
                self.summary_all_rewards.append(reward)
            print(' {}. | Episode: {} | Reward: {}'.format(self.ID, episode, reward))
            episode += 1
