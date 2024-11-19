import gym

env = gym.make("LunarLander-v2",render_mode='human') 

env.reset()



for i in range(200):
    env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)

env.close()