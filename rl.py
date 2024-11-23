import gym
from stable_baselines3 import A2C
from gym.wrappers import RecordVideo
import os

# Create output directory for videos
video_dir = "videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# First, create an environment for training (without video recording)
train_env = gym.make("LunarLander-v2")

# Train the model
print("\n=== Starting Training ===")
model1 = A2C('MlpPolicy', train_env, verbose=1)
model1.learn(total_timesteps=10000)
print("=== Training Complete ===\n")

# Create a new environment for testing (with video recording)
test_env = gym.make("LunarLander-v2", render_mode='rgb_array')
test_env = RecordVideo(
    test_env,
    video_folder=video_dir,
    name_prefix="lunar_lander",
    episode_trigger=lambda x: True,
    video_length=1000
)

# Test the trained model
episodes = 10
total_rewards = []

print("=== Starting Testing ===")
for e in range(episodes):
    print(f"\nStarting Episode {e+1}/{episodes}")
    obs, _ = test_env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    while not done:
        # Use the trained model to select actions
        action, _ = model1.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        episode_reward += reward
        steps += 1
        
        # Print step information
        print(f"Step {steps}: Reward = {reward:.2f}")
        
        if done:
            print(f"\nEpisode {e+1} finished after {steps} steps")
            print(f"Total episode reward: {episode_reward:.2f}")
            total_rewards.append(episode_reward)

print("\n=== Testing Complete ===")
print(f"Average reward per episode: {sum(total_rewards)/len(total_rewards):.2f}")
print(f"Best episode reward: {max(total_rewards):.2f}")
print(f"Worst episode reward: {min(total_rewards):.2f}")

train_env.close()
test_env.close()