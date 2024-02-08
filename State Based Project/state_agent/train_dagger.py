import torch
import numpy as np
from state_agent.utils import load_dagger_data
from state_agent.state_agent_model import Imitator, save_model, load_model
from state_agent.train_imitation import imitation_loss
import torch.utils.tensorboard as tb
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import tournament.runner as runner
from tournament.utils import StateRecorder, load_recording
import tournament.remote as remote
import geoffrey_agent
import image_jurgen_agent
import jurgen_agent
import yann_agent
import yoshua_agent


def run_match(state_file, agent1, agent2, parallel=1):
    import subprocess
    from glob import glob
    from os import path

    # Runs 5 matches of agent1 vs. agent2 with different initial ball locations
    subprocess.run(['python', '-m', 'tournament.runner', f'{agent1}', f'{agent2}', '-t', '-j', f'{parallel}', '-s', f'{state_file}'])

    trajectories = []
    path = glob(path.join('dagger_data', '*.pkl')) 
    for im_r in path:  
        rollout = load_recording(im_r)
        states = []
        for frame in rollout:
            states.append(frame)
        if len(states) > 0:
            result = states[-1]['soccer_state']['score']
            print(f'match result: {result} number of frames {len(states)}')
            trajectories.append((result, states))
    return trajectories

# Returns one training sample for DAgger
def format_dagger_data(expert_agent, player_state, opponent_state, soccer_state, team_id):
    # Ask for expert action at this frame
    expert_action = expert_agent.act(player_state, opponent_state, soccer_state)
    expert_action1 =  torch.tensor([expert_action[0]['acceleration'][0], expert_action[0]['steer'][0], expert_action[0]['brake'][0], 0], dtype=torch.float32)
    expert_action2 =  torch.tensor([expert_action[1]['acceleration'][0], expert_action[1]['steer'][0], expert_action[1]['brake'][0], 0], dtype=torch.float32)

    # Tuple of 'team1_state'
    player_team_state = (player_state, (player_state[1], player_state[0]))
    # Data format:
    # ((player_team_state1, player_team_state2), opponent_team_state, soccer_state, team_id, (expert_action1, expert_action2))
    return (player_team_state, opponent_state, soccer_state, team_id, (expert_action1, expert_action2))

def generate_dagger_data():
    training_data = []
    opponent_agent_list = ["yoshua_agent", "yann_agent", "jurgen_agent", "geoffrey_agent", "image_jurgen_agent"]
    #ball_locations = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]
    expert_agent = yoshua_agent.Team()
    expert_agent.new_match(0, 2)
    state_file = "dagger_data/temp_run.pkl"

    for opponent in opponent_agent_list:
        # State agent vs. opponent
        print(f'match in progress: state_agent vs {opponent}')
        #trajectory, result = run_match(f"dagger_data/temp_run.pkl", "state_agent", opponent, ball_location)
        match_list = run_match(state_file, "state_agent", opponent, parallel=1)
        for match in match_list:
            result, states = match
            for frame in states:
                data = format_dagger_data(expert_agent, frame['team1_state'], frame['team2_state'], frame['soccer_state'], 0)
                training_data.append(data)

        # Opponent vs. state agent
        print(f'match in progress: {opponent} vs state_agent')
        #trajectory, result = run_match(f"dagger_data/temp_run.pkl", opponent, "state_agent", ball_location)
        match_list = run_match(state_file, opponent, "state_agent", parallel=1)
        for match in match_list:
            result, states = match
            for frame in states:
                data = format_dagger_data(expert_agent, frame['team2_state'], frame['team1_state'], frame['soccer_state'], 1)
                training_data.append(data)

    print(f'Training data generated')
    return training_data

def train_dagger(args, outer_epoch):
    from os import path
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"moving model to {device}")
    model_player_1 = load_model('dagger_best_1').to(device)
    model_player_2 = load_model('dagger_best_2').to(device)

    train_logger = None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)

    max_epochs = args.inner_epoch

    optimizer_1 = optim.Adam(model_player_1.parameters(), lr=0.001, weight_decay=0.001)
    optimizer_2 = optim.Adam(model_player_2.parameters(), lr=0.001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    criterion = imitation_loss
    
    train_data = []
    for i in range(0, 1):
        train_data.extend(generate_dagger_data())
    train_data_loader = load_dagger_data(train_data)
    print(f'Dataset length = {len(train_data_loader.dataset)}')

    print(f"Starting training loop ")
    if args.log_dir is not None:
        train_logger.add_scalar('LR', optimizer_1.param_groups[0]['lr'], 0)

    best_total_avg_loss = 10000
    for epoch in range(0, max_epochs):
        model_player_1.train()
        model_player_2.train()
        model_player_1.to(device)
        model_player_2.to(device)
        print(f"Training epoch {str(epoch)}")
        total_loss = 0.0
        loss_vals1, loss_vals2 = [], []
        for state_agent_features1, state_agent_features2, expert_action1, expert_action2 in train_data_loader:
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # Move to device
            state_agent_features1, state_agent_features2 = state_agent_features1.to(device), state_agent_features2.to(device)
            expert_action1, expert_action2 = expert_action1.to(device), expert_action2.to(device)

            out_logits1 = model_player_1(state_agent_features1)
            out_logits2 = model_player_2(state_agent_features2)
            loss1 = criterion(out_logits1, expert_action1)
            loss2 = criterion(out_logits2, expert_action2)
            loss1.backward()
            loss2.backward()
            #(loss1 + loss2).backward()
            loss_vals1.append(loss1.detach().cpu().numpy())
            loss_vals2.append(loss2.detach().cpu().numpy())
            total_loss += (loss1 + loss2)
            optimizer_1.step()
            optimizer_2.step()
        
        total_avg_loss = total_loss/len(loss_vals1)
        print(f"Average loss per frame for epoch {epoch}: {total_avg_loss}")
        print(f'\tLoss 1: {sum(loss_vals1) / len(loss_vals1)}')
        print(f'\tLoss 2: {sum(loss_vals2) / len(loss_vals2)}')
        if (total_avg_loss < best_total_avg_loss):
            print(f'\tSaving new best model at epoch {epoch}')
            save_model(model_player_1, f"dagger_best_1")
            save_model(model_player_2, f"dagger_best_2")
            save_model(model_player_1, f"dagger_best_1_{outer_epoch}")
            save_model(model_player_2, f"dagger_best_2_{outer_epoch}")
            best_total_avg_loss = total_avg_loss
    save_model(model_player_1, f"dagger_1_{outer_epoch}")
    save_model(model_player_2, f"dagger_2_{outer_epoch}")

def train_dagger_outer(args):
    for epoch in range(0, args.outer_epoch):
        print(f'Starting outer epoch {epoch}...')
        train_dagger(args, outer_epoch=epoch)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir')
    # Put custom arguments here
    parser.add_argument('-i', '--inner_epoch', type=int, default=20)
    parser.add_argument('-o', '--outer_epoch', type=int, default=10)
    args = parser.parse_args()
    train_dagger_outer(args)