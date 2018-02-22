
import gym
from baselines import deepq
import tensorflow as tf

episode_rewards = []

def callback(lcl, glb):
    global episode_rewards
    episode_rewards = lcl['episode_rewards']
    if len(episode_rewards) > 2000:
        return True
    else:
        return False

def main():
    global episode_rewards
    env = gym.make("CartPole-v1")
    max_timesteps_env = env.env._spec.__dict__['tags']['wrapper_config.TimeLimit.max_episode_steps']
    model = deepq.models.mlp([64], activation_fn=tf.nn.tanh)
    act = deepq.learn(
        env,
        lr=1e-08,
        q_func=model,
        target_network_update_freq=1,
        batch_size=32,
        max_timesteps=max_timesteps_env*2000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print('done')
    return episode_rewards
    # print("Saving model to cartpole_model.pkl")
    # act.save("cartpole_model.pkl")


if __name__ == '__main__':
    print(main())
    print('^^^^^^^^^^^^^')
