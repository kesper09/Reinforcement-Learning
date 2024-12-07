import gymnasium as gym
from stable_baselines3 import PPO, A2C
import os
import glob
import re
from datetime import datetime

# Create base directories
MODEL_DIRS_A2C = "models/A2C"
MODEL_DIRS_PPO = "models/PPO"
LOGSDIR = "logs"

# Ensure directories exist
for directory in [MODEL_DIRS_A2C, MODEL_DIRS_PPO, LOGSDIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_next_run_number(model_dir):
    """Get the next sequential run number for the model directory."""
    existing_models = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
    return len(existing_models) + 1

def find_latest_model_file(model_dir):
    """Find the latest model file in the given directory."""
    # Look for .zip files in the directory and its subdirectories
    model_files = glob.glob(os.path.join(model_dir, '**/*.zip'), recursive=True)
    
    if not model_files:
        return None
    
    # Return the most recently created model file
    return max(model_files, key=os.path.getctime)

def extract_run_number(run_name):
    """Extract run number from run directory name."""
    match = re.search(r'Run(\d+)', run_name)
    return int(match.group(1)) if match else 1

def load_or_create_model(env, model_type, is_load_existing):
    """Load an existing model or create a new one based on user preference."""
    model_dirs = MODEL_DIRS_A2C if model_type == 1 else MODEL_DIRS_PPO
    model_class = A2C if model_type == 1 else PPO

    if is_load_existing:
        # List available model runs
        existing_models = [f for f in os.listdir(model_dirs) if os.path.isdir(os.path.join(model_dirs, f))]
        
        if not existing_models:
            custom_path = input("No existing models found. Enter a custom model path (or press Enter to start fresh): ").strip()
            
            if custom_path and os.path.exists(custom_path):
                print(f"Loading model from custom path: {custom_path}")
                return model_class.load(custom_path, env=env), get_next_run_number(model_dirs)
            
            print("Creating a new model.")
            return model_class('MlpPolicy', env, verbose=1, tensorboard_log=LOGSDIR), 1
        
        print("Available model runs:")
        for idx, model_name in enumerate(existing_models, 1):
            print(f"{idx}. {model_name}")
        
        model_choice = int(input("Enter the number of the model run to load: ")) - 1
        model_run_dir = os.path.join(model_dirs, existing_models[model_choice])
        
        # Find the latest model file in the selected run directory
        latest_model_path = find_latest_model_file(model_run_dir)
        
        if latest_model_path:
            print(f"Loading model from {latest_model_path}")
            loaded_model = model_class.load(latest_model_path, env=env)
            return loaded_model, extract_run_number(existing_models[model_choice])
        
        print("No model file found in the selected run. Creating a new model.")
        return model_class('MlpPolicy', env, verbose=1, tensorboard_log=LOGSDIR), 1
    else:
        # Create a new model
        run_number = get_next_run_number(model_dirs)
        return model_class('MlpPolicy', env, verbose=1, tensorboard_log=LOGSDIR), run_number

def main():
    # First, create an environment for training
    train_env = gym.make("LunarLander-v3")

    # Setting a timestep for training the model
    TIMESTEPS = 3000

    # Get user input for model selection
    mod = int(input("Enter the model that you would like to run: 1-> A2C 2-> PPO: "))
    is_load_existing = int(input("Load existing model? 1->Yes 0->No: ")) == 1

    # Determine model directories and class
    model_dirs = MODEL_DIRS_A2C if mod == 1 else MODEL_DIRS_PPO

    # Load or create model
    model, run_number = load_or_create_model(train_env, mod, is_load_existing)

    # Prepare logging configuration
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"{'A2C' if mod == 1 else 'PPO'}_{current_time}_Run{run_number}"

    # Train the model
    for i in range(1, 30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
        
        # Create a unique directory for each run number
        run_dir = os.path.join(model_dirs, f"Run{run_number}")
        os.makedirs(run_dir, exist_ok=True)
        
        model.save(f"{run_dir}/{TIMESTEPS*i}")
        print(f"Step: {i}")

    # Close the environment after training
    train_env.close()

if __name__ == "__main__":
    main()