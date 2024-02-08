import torch
import numpy as np

import geoffrey_agent
import image_jurgen_agent
import jurgen_agent
import yann_agent
import yoshua_agent
from state_agent.utils import load_reinforce_data
from state_agent.state_agent_model import Imitator, save_model, load_model
from state_agent.train_imitation import imitation_loss
import torch.utils.tensorboard as tb
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import tournament.runner as runner
from tournament.utils import StateRecorder, load_recording
import tournament.remote as remote
from state_agent.player import Team
from torch.distributions import Bernoulli, Normal


def run_match(state_file, agent1, agent2, parallel=1):
    import subprocess
    from glob import glob
    from os import path

    # Runs 5 matches of agent1 vs. agent2 with different initial ball locations
    subprocess.run(['python', '-m', 'tournament.runner', f'{agent1}', f'{agent2}', '-t', '-j', f'{parallel}', '-s', f'{state_file}'])

    trajectories = []
    path = glob(path.join('reinforce_data', '*.pkl')) 
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

def format_reinforce_data(player_state, opponent_state, soccer_state, team_id, actions):
    state = (player_state, opponent_state, soccer_state, team_id)
    #0 team1 p1
    #1 team2 p1
    #2 team1 p2
    #3 team2 p2
    action1 = torch.tensor([actions[team_id]['acceleration'] > 0.5, actions[team_id]['steer'], actions[team_id]['brake'], 0], dtype=torch.float32)
    action2 = torch.tensor([actions[team_id+2]['acceleration'] > 0.5, actions[team_id+2]['steer'], actions[team_id+2]['brake'], 0], dtype=torch.float32)
    return state, (action1, action2)

def calculate_puck_to_goal_line(soccer_state, team_id):
    # Calculate reward
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
    puck_to_goal_line = (goal_line_center - puck_center)
    puck_to_goal_line = torch.norm(puck_to_goal_line)
    return puck_to_goal_line

def generate_reinforce_data():
    opponent_agent_list = ["yoshua_agent"]#["yoshua_agent", "yann_agent", "jurgen_agent", "geoffrey_agent", "image_jurgen_agent"]
    state_list = []  #tuple (state1, state2)
    action_list = [] #tuple (p1, p2) 
    reward_list = [] #just one value for now
    state_file = "reinforce_data/temp_run.pkl"

    for opponent in opponent_agent_list:
        print(f'match in progress: state_agent vs {opponent}')
        match_list = run_match(state_file, "state_agent", opponent)
        for match in match_list:
            result, states = match
            #print( torch.tensor(states[0]['soccer_state']['ball']['location'], dtype=torch.float32)[[0, 2]])
            total_reward = 0
            
            #base_reward = (result[0] - result[1])
            #base_reward += (result[0] - result[1])/len(states)
            
            for frame in states:
                team_id = 0
                state, action = format_reinforce_data(frame['team1_state'], frame['team2_state'], frame['soccer_state'], team_id, frame['actions'])
                kart_position = torch.tensor(frame['team1_state'][0]['kart']['location'], dtype=torch.float32)[[0, 2]]
                puck_center = torch.tensor(frame['soccer_state']['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
                #puck_center = torch.tensor([0,0])
                #puck_to_goal_line = calculate_puck_to_goal_line(frame['soccer_state'], team_id)
                #puck_to_goal_line /= len(states)
                # consider overall reward for keeping puck on other side?
                #current_reward = base_reward - puck_to_goal_line
                puck_to_kart = (kart_position[0] - puck_center)
                puck_to_kart = torch.norm(puck_to_kart)
                puck_to_kart/= len(states)
                current_reward = puck_to_kart
                total_reward +=current_reward

                state_list.append(state)
                action_list.append(action)
                reward_list.append(current_reward)
            '''
            for i in range(len(states)):
                team_id = 0
                state, action = format_reinforce_data(states[i]['team1_state'], states[i]['team2_state'], states[i]['soccer_state'], team_id, states[i]['actions'])
                # Reward decreasing distance from puck to goal_line
                prev_puck_to_goal_line = calculate_puck_to_goal_line(states[min(0, i-1)]['soccer_state'], team_id)
                cur_puck_to_goal_line = calculate_puck_to_goal_line(states[i]['soccer_state'], team_id)
                puck_reward = (prev_puck_to_goal_line - cur_puck_to_goal_line) / len(states)
                # Reward scoring at that frame (and frames moving forward)
                #score_reward = (states[i]['soccer_state']['score'][0]/len(states)) - states[i]['soccer_state']['score'][1]
                current_reward = base_reward + puck_reward
                total_reward += current_reward

                state_list.append(state)
                action_list.append(action)
                reward_list.append(current_reward)
            '''
            print(f'match result: {result} number of frames {len(states)}\nbase reward: {0} \ntotal reward: {total_reward}')
            
        print(f'match in progress: {opponent} vs state_agent')
        match_list = run_match(state_file, opponent, "state_agent")
        for match in match_list:
            result, states = match
            total_reward = 0
            
            #base_reward = (result[1] - result[0])
            #base_reward += (result[1] - result[0])/len(states)
            
            for frame in states:
                team_id = 1
                state, action = format_reinforce_data(frame['team2_state'], frame['team1_state'], frame['soccer_state'], team_id, frame['actions'])
                kart_position = torch.tensor(frame['team2_state'][0]['kart']['location'], dtype=torch.float32)[[0, 2]]
                puck_center = torch.tensor(frame['soccer_state']['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
                #puck_center = torch.tensor([0,0])
                #puck_to_goal_line = calculate_puck_to_goal_line(frame['soccer_state'], team_id)
                # puck_to_goal_line /= len(states)
                # consider overall reward for keeping puck on other side?
                #current_reward = base_reward - puck_to_goal_line
                puck_to_kart = (kart_position[0] - puck_center)
                puck_to_kart = torch.norm(puck_to_kart)
                puck_to_kart/= len(states)
                current_reward = puck_to_kart
                total_reward += current_reward

                state_list.append(state)
                action_list.append(action)
                reward_list.append(current_reward)
            '''
            for i in range(len(states)):
                team_id = 1
                state, action = format_reinforce_data(states[i]['team2_state'], states[i]['team1_state'], states[i]['soccer_state'], team_id, states[i]['actions'])
                # Reward decreasing distance from puck to goal_line
                prev_puck_to_goal_line = calculate_puck_to_goal_line(states[min(0, i-1)]['soccer_state'], team_id)
                cur_puck_to_goal_line = calculate_puck_to_goal_line(states[i]['soccer_state'], team_id)
                puck_reward = (prev_puck_to_goal_line - cur_puck_to_goal_line) / len(states)
                # Reward scoring at that frame (and frames moving forward)
                #score_reward = (states[i]['soccer_state']['score'][1]/len(states)) - states[i]['soccer_state']['score'][0]
                current_reward = base_reward + puck_reward
                total_reward += current_reward

                state_list.append(state)
                action_list.append(action)
                reward_list.append(current_reward)
            '''
            print(f'match result: {result} number of frames {len(states)}\nbase reward: {0} \ntotal reward: {total_reward}')
            
    reward_tensor = torch.tensor(reward_list, dtype=torch.float32)
    mean = torch.mean(reward_tensor)
    std_dev = torch.std(reward_tensor)
    print(f"rewards mean: {mean} std dev: {std_dev}")
    normalized_reward = ((reward_tensor - mean) / std_dev).tolist()

    #training_samples = [(state[i], action[i], reward[i]) for i in range(0, len(rewards))]
    """
    Element of training_samples:
    ((current_state_1, current_state_2, soccer_state, team_id),(current_action_1, current_action_2),reward)
    """
    training_samples = zip(state_list, action_list, normalized_reward)
    print(f'Training data generated')
    return training_samples 

def train_reinforce(args, outer_epoch):
    from os import path
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"moving model to {device}")
    model = load_model("imitation_cindy")
    print(model)
    train_logger = None, None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)

    max_epochs = args.inner_epoch

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    
    train_data = []
    for i in range(0, 1):
        train_data.extend(generate_reinforce_data())
    train_data_loader = load_reinforce_data(train_data)
  
    print(f'Dataset length = {len(train_data_loader.dataset)}')

    print(f"Starting training loop ")
    if args.log_dir is not None:
        train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], 0)

    best_total_avg_reward = float('-inf')
    for epoch in range(0, max_epochs):
        model.train()
        model = model.to(device)
        print(f"Training epoch {str(epoch)}")
        #total_reward = 0.0
        reward_vals = []
    #note: if buggy, try parentheses
        for state_agent_features, action1, action2, reward in train_data_loader:
            optimizer.zero_grad()

            # Move to device
            state_agent_features= state_agent_features.to(device)
            reward = reward.to(device)
            action1, action2 = action1.to(device), action2.to(device)

            out_logits = model(state_agent_features) #B:10

            pi_accel_1 = Bernoulli(logits=out_logits[:,0]) #BNorms
            pi_steer_1 = Normal(torch.clamp(out_logits[:,1], min=-1.0, max=1.0), torch.nn.functional.softplus(out_logits[:,2]) + 0.1)
            pi_brake_1 = Bernoulli(logits=out_logits[:,3])
            pi_drift_1 = Bernoulli(logits=out_logits[:,4])
            pi_accel_2 = Bernoulli(logits=out_logits[:,5])
            pi_steer_2 = Normal(torch.clamp(out_logits[:,6], min=-1.0, max=1.0), torch.nn.functional.softplus(out_logits[:,7]) + 0.1)
            pi_brake_2 = Bernoulli(logits=out_logits[:,8])
            pi_drift_2 = Bernoulli(logits=out_logits[:,9])
            


            # PPO, multi-agent DDPG, soft actor critic. Hugging Face


            expected_log_accel_1 = (pi_accel_1.log_prob(action1[:,0])*reward).mean()
            expected_log_accel_2 = (pi_accel_2.log_prob(action2[:,0])*reward).mean()
            expected_log_steer_1 = (pi_steer_1.log_prob(action1[:,1])*reward).mean()
            expected_log_steer_2 = (pi_steer_2.log_prob(action2[:,1])*reward).mean()
            expected_log_brake_1 = (pi_brake_1.log_prob(action1[:,2])*reward).mean()
            expected_log_brake_2 = (pi_brake_2.log_prob(action2[:,2])*reward).mean()
            expected_log_drift_1 = (pi_drift_1.log_prob(action1[:,3])*reward).mean()
            expected_log_drift_2 = (pi_drift_2.log_prob(action2[:,3])*reward).mean()

            print(f'reward {reward} steer_mean: {torch.clamp(out_logits[0][1], min=-1.0, max=1.0)} steer_var: {torch.nn.functional.softplus(out_logits[0][2]) + 0.1} raw_mean: {out_logits[0][1]} raw_var:{out_logits[0][2]} sample: {pi_steer_1.sample()[0]}')
            expected_log_full = (expected_log_accel_1 + expected_log_accel_2 + expected_log_steer_1 + expected_log_steer_2
                                 + expected_log_brake_1 + expected_log_brake_2 + expected_log_drift_1 + expected_log_drift_2)/8

            #print(f'\tEXPECTED_LOG_FULL: {expected_log_full}\n')
            (-expected_log_full).backward()
            
            reward_vals.append(expected_log_full.detach().cpu().numpy())
            #total_reward += expected_log_full.detach().cpu().numpy()
            optimizer.step()

        total_avg_reward = sum(reward_vals)/len(reward_vals)
        print(f"Average reward per frame for epoch {epoch}: {total_avg_reward}")
        if (total_avg_reward > best_total_avg_reward):
            print(f'\tSaving new best model at epoch {epoch}')
            save_model(model, f"reinforce_best")
            save_model(model, f"reinforce_best_{outer_epoch}")
            best_total_avg_reward = total_avg_reward
    save_model(model, f"reinforce_{outer_epoch}")

def train_reinforce_outer(args):
    for epoch in range(0, args.outer_epoch):
        print(f'Starting outer epoch {epoch}...')
        train_reinforce(args, outer_epoch=epoch)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir')
    # Put custom arguments here
    parser.add_argument('-i', '--inner_epoch', type=int, default=25)
    parser.add_argument('-o', '--outer_epoch', type=int, default=10)
    args = parser.parse_args()
    train_reinforce_outer(args)