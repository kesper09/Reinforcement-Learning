import os
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import torch
import json

def get_run_number(model_type):
    """
    Get the next run number for the specified model type.
    Creates a tracking file if it doesn't exist.
    """
    model_name = "A2C" if model_type == 1 else "PPO"
    tracking_file = os.path.join("models", f"{model_name}_runs.json")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Read or initialize run tracking
    try:
        with open(tracking_file, 'r') as f:
            run_tracking = json.load(f)
    except FileNotFoundError:
        run_tracking = {"current_run": 1}
    
    # Increment run number
    run_tracking["current_run"] += 1
    
    # Save updated run tracking
    with open(tracking_file, 'w') as f:
        json.dump(run_tracking, f)
    
    return run_tracking["current_run"]

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

def extract_run_number(model_path):
    """Extract the run number from the model path."""
    parts = model_path.split(os.path.sep)
    # Look for the part that contains 'Run'
    for part in parts:
        if part.startswith('Run'):
            try:
                return int(part.split('Run')[1])
            except:
                break
    return 1  # Default to 1 if no run number found

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
    
    # Extract run number from the model path
    run_number = extract_run_number(model_path)
    
    # Prepare environment
    base_video_path = "/home/malice/Documents/Codes/Reinforcement learning/videos"
    model_name = "A2C" if model_type == 1 else "PPO"
    
    # Create parent directory for model type
    model_video_dir = os.path.join(base_video_path, model_name)
    os.makedirs(model_video_dir, exist_ok=True)
    
    # Create run-specific directory
    video_path = os.path.join(model_video_dir, f"Run_{run_number}")
    os.makedirs(video_path, exist_ok=True)

    print(f"Video will be saved to: {video_path}")
    
    # Create a new environment for testing (with video recording)
    test_env = gym.make("LunarLander-v3", render_mode='rgb_array')
    test_env = RecordVideo(
        test_env,
        video_folder=video_path,
        name_prefix="lunar_lander",
        episode_trigger=lambda x: True
    )
    
    # Load model
    model_class = A2C if model_type == 1 else PPO
    model = model_class.load(model_path, device=device)

    # Test the model
    num_episodes = 10
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
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"\nEpisode {e+1} finished after {steps} steps")
                print(f"Total episode reward: {episode_reward:.2f}")
                total_rewards.append(episode_reward)
    
    print("\n=== Testing Complete ===")
    print(f"Number of episodes: {len(total_rewards)}")
    print(f"Average reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Best episode reward: {max(total_rewards):.2f}")
    print(f"Worst episode reward: {min(total_rewards):.2f}")
    
    test_env.close()

if __name__ == "__main__":
    main()