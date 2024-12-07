import gymnasium as gym
from stable_baselines3 import A2C
from gymnasium.wrappers import RecordVideo
import os

# Get user input
timesteps = int(input("Enter the number of timesteps for training (default 10000): ") or "10000")
num_episodes = int(input("Enter the number of test episodes (default 10): ") or "10")

# Create output directory for videos
video_dir = "videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# First, create an environment for training (without video recording)
train_env = gym.make("LunarLander-v3", render_mode=None)

# Train the model
print(f"\n=== Starting Training with {timesteps} timesteps ===")
model1 = A2C('MlpPolicy', train_env, verbose=1)
model1.learn(total_timesteps=timesteps)
print("=== Training Complete ===\n")

# Create a new environment for testing (with video recording)
test_env = gym.make("LunarLander-v3", render_mode='rgb_array')
test_env = RecordVideo(
    test_env,
    video_folder=video_dir,
    name_prefix="lunar_lander",
    episode_trigger=lambda x: True
)

# Test the trained model
total_rewards = []

print(f"=== Starting Testing with {num_episodes} episodes ===")
for e in range(num_episodes):
    print(f"\nStarting Episode {e+1}/{num_episodes}")
    obs, info = test_env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    steps = 0
    
    while not (terminated or truncated):
        # Use the trained model to select actions
        action, _ = model1.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward
        steps += 1
        
        # Print step information
        print(f"Step {steps}: Reward = {reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode {e+1} finished after {steps} steps")
            print(f"Total episode reward: {episode_reward:.2f}")
            total_rewards.append(episode_reward)

print("\n=== Testing Complete ===")
print(f"Average reward per episode: {sum(total_rewards)/len(total_rewards):.2f}")
print(f"Best episode reward: {max(total_rewards):.2f}")
print(f"Worst episode reward: {min(total_rewards):.2f}")

train_env.close()
test_env.close()