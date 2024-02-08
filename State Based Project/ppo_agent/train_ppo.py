import sys
import torch

from ppo_agent.arguments import get_args
from ppo_agent.player import Team
from ppo_agent.model import FeedForwardNN


def train(hyperparameters):
    print(f"Training", flush=True)
    # Create a model for PPO.
    model = Team(policy_class=FeedForwardNN, training_mode=True, load_training_model=False, **hyperparameters)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=1000000000)
def main(args):
    hyperparameters = {
                'timesteps_per_batch': 12000,
                'max_timesteps_per_episode': 1200,
                'gamma': 0.99,
                'n_updates_per_iteration': 50,
                'lr': 1e-3,
                'clip': 0.2,
              }

    train(hyperparameters=hyperparameters)



if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)