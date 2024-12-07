import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import torch

def get_latest_models(model_type):
    """Find latest 5 models of specified type."""
    model_dir = "models/A2C" if model_type == 1 else "models/PPO"
    
    # Find all .zip files recursively
    model_files = []
    for root, _, files in os.walk(model_dir):
        model_files.extend([os.path.join(root, f) for f in files if f.endswith('.zip')])
    
    # Sort by modification time, most recent first
    model_files.sort(key=os.path.getmtime, reverse=True)
    
    return model_files[:5]

def main():
    # Force CPU usage
    device = torch.device("cpu")

    # Select model type
    model_type = int(input("Select model type: 1-> A2C, 2-> PPO: "))
    
    # Get latest models
    latest_models = get_latest_models(model_type)
    
    # Display and select model
    if not latest_models:
        print("No models found.")
        return
    
    print("Latest models:")
    for i, model_path in enumerate(latest_models, 1):
        print(f"{i}. {model_path}")
    
    # Default to first (latest) model
    choice = input(f"Select model (default=1): ").strip() or '1'
    model_path = latest_models[int(choice) - 1]
    
    # Prepare environment
    env = gym.make("LunarLander-v3", render_mode="human")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Load model
    model_class = A2C if model_type == 1 else PPO
    model = model_class.load(model_path, env=env, device=device)

    # Run episodes
    episodes = 10
    for i in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            done = done[0]
            env.render()
        print(f"Episode {i+1} completed")

    env.close()

if __name__ == "__main__":
    main()