import torch
import numpy as np
import yoshua_agent
import geoffrey_agent
import image_jurgen_agent
import jurgen_agent
import yann_agent
import yoshua_agent
from state_agent.utils import load_imitation_data, load_dagger_data, load_reinforce_data, extract_features
from state_agent.state_agent_model import Imitator, save_model, load_model
import torch.utils.tensorboard as tb
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import tournament.runner as runner
from tournament.utils import StateRecorder, load_recording
import tournament.remote as remote
from state_agent.player import Team
from torch.distributions import Bernoulli, Normal


def imitation_loss(logits, expert_actions):
    # Extract continuous and discrete parts from logits
    mean_acc = logits[:, 0]
    var_acc = logits[:, 1]
    mean_steer = logits[:, 2]
    var_steer = logits[:, 3]
    discrete_logits = logits[:, 4:6]

    # Ensure variances are positive (e.g., by using softplus)
    var_acc = torch.nn.functional.softplus(var_acc)
    var_steer = torch.nn.functional.softplus(var_steer)

    # Continuous expert actions
    continuous_expert_actions = expert_actions[:, :2]

    # Discrete expert actions
    discrete_expert_actions = expert_actions[:, 2:]

    # Loss for continuous actions
    gaussian_nll_loss = nn.GaussianNLLLoss()
    loss_continuous_acc = gaussian_nll_loss(mean_acc, continuous_expert_actions[:, 0], var_acc)
    loss_continuous_steer = gaussian_nll_loss(mean_steer, continuous_expert_actions[:, 1], var_steer)
    loss_continuous = loss_continuous_acc + loss_continuous_steer

    # Loss for discrete actions
    bce_with_logits_loss = nn.BCEWithLogitsLoss()
    loss_discrete = bce_with_logits_loss(discrete_logits, discrete_expert_actions)

    # Combine the losses
    total_loss = loss_continuous + loss_discrete
    return total_loss

def train_imitation(args):
    from os import path
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = Imitator(input_size=56)
    print(f"{model}")
    print(f"moving model to {device}")
    model = model.to(device)

    train_logger = None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)

    scheduler_enabled = True
    print("Started Loading data")
    train_dataset_loader = load_imitation_data('imitation_data', max_pkls=100)
    print("Finished loading data")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    criterion = imitation_loss

    dataset_size = len(train_dataset_loader.dataset)
    print(f"Starting training loop for dataset_size: {dataset_size}")
    if scheduler_enabled and train_logger is not None:
        train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], 0)

    max_epochs = args.epoch
    for epoch in range(0, max_epochs):
        model.train()
        print(f"Training epoch {str(epoch)}")
        i = 0
        loss_vals = []
        for batch_x, batch_y in train_dataset_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logit = model.forward(batch_x)
            # print("forward pass finished")
            #print(logit.shape)
            #print(batch_y.shape)
            loss = criterion(logit, batch_y)
            loss_vals.append(loss.detach().cpu().numpy())

            if train_logger is not None:
                train_logger.add_scalar('loss', float(loss), epoch * dataset_size + i)
            loss.backward()
            optimizer.step()
            i += 1
        avg_loss = sum(loss_vals) / len(loss_vals) 
        print(f'\tloss = {avg_loss}')

        if scheduler_enabled:
            scheduler.step()  # val_acc)
            if train_logger is not None:
                train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch * dataset_size + i - 1)
        save_model(model, f"imitation_{epoch}")
    save_model(model, "imitation_final")

def run_match(state_file, team1, team2, ball_location, parallel):
    from os import environ
    from pathlib import Path
    import logging

    record_state = state_file
    num_players = 2
    num_frames = 1200
    max_score = 3
    result = None

    logging.basicConfig(level=environ.get('LOGLEVEL', 'WARNING').upper())

    if parallel == 0 or remote.ray is None:
        # Create the teams
        team1 = runner.AIRunner() if team1 == 'AI' else runner.TeamRunner(team1)
        team2 = runner.AIRunner() if team2 == 'AI' else runner.TeamRunner(team2)

        # What should we record?
        recorder = None
        if record_state:
            recorder = recorder & StateRecorder(record_state)

        # Start the match
        match = runner.Match(use_graphics=team1.agent_type == 'image' or team2.agent_type == 'image')
        try:
            result = match.run(team1, team2, num_players, num_frames, max_score=max_score, initial_ball_location=ball_location, record_fn=recorder)
        except runner.MatchException as e:
            print('Match failed', e.score)
            print(' T1:', e.msg1)
            print(' T2:', e.msg2)

        rollout = load_recording(record_state)
        states = []
        for frame in rollout:
            states.append(frame)
    else:
        print('Starting parallel matches...')
        # Fire up ray
        remote.init(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()), configure_logging=True,
                    log_to_driver=True, include_dashboard=False)

        # Create the teams
        team1 = runner.AIRunner() if team1 == 'AI' else remote.RayTeamRunner.remote(team1)
        team2 = runner.AIRunner() if team2 == 'AI' else remote.RayTeamRunner.remote(team2)
        team1_type, *_ = team1.info() if team1 == 'AI' else remote.get(team1.info.remote())
        team2_type, *_ = team2.info() if team2 == 'AI' else remote.get(team2.info.remote())

        # Start the match
        results = []
        for i in range(parallel):
            recorder = None
            if record_state:
                ext = Path(record_state).suffix
                recorder = remote.RayStateRecorder.remote(record_state.replace(ext, f'.{i}{ext}'))

            match = remote.RayMatch.remote(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()),
                                           use_graphics=team1_type == 'image' or team2_type == 'image')
            result = match.run.remote(team1, team2, num_players, num_frames, max_score=max_score,
                                      initial_ball_location=ball_location,
                                      record_fn=recorder)
            results.append(result)

        for result in results:
            try:
                result = remote.get(result)
            except (remote.RayMatchException, runner.MatchException) as e:
                print('Match failed', e.score)
                print(' T1:', e.msg1)
                print(' T2:', e.msg2)

            print('Match results', result)
        return results
    return states, result

# Returns one training sample for DAgger
def format_dagger_data(expert_agent, player_state, opponent_state, soccer_state, team_id):
    # Ask for expert action at this frame
    expert_action = expert_agent.act(player_state, opponent_state, soccer_state)
    expert_action1 =  torch.tensor([expert_action[0]['acceleration'][0], expert_action[0]['steer'][0], expert_action[0]['brake'][0], 0], dtype=torch.float32)
    expert_action2 =  torch.tensor([expert_action[1]['acceleration'][0], expert_action[1]['steer'][0], expert_action[1]['brake'][0], 0], dtype=torch.float32)

    state1 = (player_state, opponent_state, soccer_state)
    state2 = ((player_state[1], player_state[0]), opponent_state, soccer_state)
    # Tuple of tuples: 
    # data[0] = ((p1_state, opponent_state, soccer_state), expert_action)
    # data[1] = ((p2_state, opponent_state, soccer_state), expert_action)
    # data[2] = team_id
    return ((state1, expert_action1), (state2, expert_action2), team_id)

def generate_dagger_data():
    from os import remove

    training_data = []
    opponent_agent_list = ["yoshua_agent", "yann_agent", "jurgen_agent", "geoffrey_agent", "image_jurgen_agent"]
    ball_locations = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]
    expert_agent = yoshua_agent.Team()
    expert_agent.new_match(0, 2)
    for opponent in opponent_agent_list:
        for ball_location in ball_locations:
            # State agent vs. opponent
            print(f'match in progress: state_agent vs {opponent}')
            trajectory, result = run_match(f"dagger_data/temp_run.pkl", "state_agent", opponent, ball_location)
            print(f'match result: {result} number of frames {len(trajectory)}')
            for frame in trajectory:
                data = format_dagger_data(expert_agent, frame['team1_state'], frame['team2_state'], frame['soccer_state'], 0)
                training_data.append(data)
            remove("dagger_data/temp_run.pkl")

            # Opponent vs. state agent
            print(f'match in progress: {opponent} vs state_agent')
            trajectory, result = run_match(f"dagger_data/temp_run.pkl", opponent, "state_agent", ball_location)
            print(f'match result: {result} number of frames {len(trajectory)}')
            for frame in trajectory:
                data = format_dagger_data(expert_agent, frame['team2_state'], frame['team1_state'], frame['soccer_state'], 1)
                training_data.append(data)
            remove("dagger_data/temp_run.pkl")
    print(f'Training data generated')
    return training_data

def train_dagger(args, initial, outer_epoch):
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

    train_logger = None, None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)


    max_epochs = 30

    optimizer_1, optimizer_2 = optim.Adam(model_player_1.parameters(), lr=0.001, weight_decay=0.001), optim.Adam(model_player_2.parameters(), lr=0.001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
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
        save_model(model_player_1, f"dagger_1_{outer_epoch}_{epoch}")
        save_model(model_player_2, f"dagger_2_{outer_epoch}_{epoch}")

def generate_reinforce_data():
    from os import remove

    opponent_agent_list = ["yoshua_agent", "yann_agent", "jurgen_agent",]#["yoshua_agent", "yann_agent", "jurgen_agent", "geoffrey_agent", "image_jurgen_agent"]
    ball_locations = [[0,0]]#[[0,0], [1,0], [-1,0], [0,1], [0,-1]]
    expert_agent = Team()
    expert_agent.new_match(0, 2)
    state = []  #tuple (state1, state2)
    action = [] #tuple (p1, p2) 
    reward = [] #just one value for now
    for opponent in opponent_agent_list:
        for ball_location in ball_locations:
          print(f'match in progress: state_agent vs {opponent}')
          trajectory, result = run_match(f"reinforce_data/temp_run.pkl", "state_agent", opponent, ball_location)

          base_reward = (result[0] - result[1])
          base_reward += (result[0] - result[1])/len(trajectory)

          total_reward = 0
          for frame in trajectory:
            # Ask for expert action at this frame
            #expert_action = expert_agent.act(frame['team1_state'], frame['team2_state'], frame['soccer_state'])
            current_state_1 = frame['team1_state']
            current_state_2 = frame['team2_state']
            soccer_state = frame['soccer_state']
            current_action = frame['actions']
            team_id = 0
            current_action_1 =  torch.tensor([current_action[0]['acceleration'], current_action[0]['steer'], current_action[0]['brake'], 0], dtype=torch.float32)
            current_action_2 =  torch.tensor([current_action[2]['acceleration'], current_action[2]['steer'], current_action[2]['brake'], 0], dtype=torch.float32)
            #0 p1 team1
            #1 p1 team2
            #2 p2 team1
            puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
            goal_line_center = torch.tensor(soccer_state['goal_line'][0], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
            puck_to_goal_line = (goal_line_center - puck_center)
            puck_to_goal_line = torch.norm(puck_to_goal_line)/len(trajectory)

            # consider overall reward for keeping puck on other side?
            current_reward = base_reward - puck_to_goal_line
            total_reward += current_reward
            state.append((current_state_1, current_state_2, soccer_state, team_id))
            action.append((current_action_1, current_action_2))
            reward.append(current_reward)
          print(f'match result: {result} number of frames {len(trajectory)}\nbase reward: {base_reward} \ntotal reward: {total_reward}')
          print(f'match in progress: {opponent} vs state_agent')
          remove("reinforce_data/temp_run.pkl")
          trajectory, result = run_match(f"reinforce_data/temp_run.pkl", opponent, "state_agent", ball_location)
          base_reward = (result[1] - result[0])
          base_reward += (result[1] - result[0])/len(trajectory)
          for frame in trajectory:
            current_state_1 = frame['team1_state']
            current_state_2 = frame['team2_state']
            soccer_state = frame['soccer_state']
            current_action = frame['actions']
            team_id = 1
            current_action_1 =  torch.tensor([current_action[1]['acceleration'], current_action[1]['steer'], current_action[1]['brake'], 0], dtype=torch.float32)
            current_action_2 =  torch.tensor([current_action[3]['acceleration'], current_action[3]['steer'], current_action[3]['brake'], 0], dtype=torch.float32)
            #0 p1 team1
            #1 p1 team2
            #2 p2 team1
            puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
            goal_line_center = torch.tensor(soccer_state['goal_line'][(1)], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
            puck_to_goal_line = (goal_line_center - puck_center)
            puck_to_goal_line = torch.norm(puck_to_goal_line)/len(trajectory)

            current_reward = base_reward - puck_to_goal_line
            reward.append(current_reward)

            state.append((current_state_1, current_state_2, soccer_state, team_id))
            action.append((current_action_1, current_action_2))
            reward.append(current_reward)
          print(f'match result: {result} number of frames {len(trajectory)}\nbase reward: {base_reward} \ntotal reward: {total_reward}')
          remove("reinforce_data/temp_run.pkl")
    reward_tensor = torch.tensor(reward, dtype=torch.float32)
    mean = torch.mean(reward_tensor)
    std_dev = torch.std(reward_tensor)
    print(f"rewards mean: {mean} std dev: {std_dev}")
    normalized_reward = ((reward_tensor - mean) / std_dev).tolist()

    #training_samples = [(state[i], action[i], reward[i]) for i in range(0, len(rewards))]
    """
    Element of training_samples:
    ((current_state_1, current_state_2, soccer_state, team_id),(current_action_1, current_action_2),reward)
    """
    training_samples = zip(state, action, normalized_reward)
    print(f'Training data generated')
    return training_samples

def train_reinforce(args, initial, outer_epoch):
    from os import path
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"moving model to {device}")
    model_player_1 = load_model('reinforce_best_1').to(device)
    model_player_2 = load_model('reinforce_best_2').to(device)
    print(model_player_2)
    train_logger = None, None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)

    max_epochs = 25

    optimizer_1, optimizer_2 = optim.Adam(model_player_1.parameters(), lr=0.0001, weight_decay=0.001), optim.Adam(model_player_2.parameters(), lr=0.0001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    criterion = imitation_loss
    
    train_data = []
    for i in range(0, 1):
        train_data.extend(generate_reinforce_data())
    train_data_loader = load_reinforce_data(train_data)
  
    print(f'Dataset length = {len(train_data_loader.dataset)}')

    print(f"Starting training loop ")
    if args.log_dir is not None:
        train_logger.add_scalar('LR', optimizer_1.param_groups[0]['lr'], 0)

    best_total_avg_reward = float('-inf')
    for epoch in range(0, max_epochs):
        model_player_1.train()
        model_player_2.train()
        model_player_1 = model_player_1.to(device)
        model_player_2 = model_player_2.to(device)
        print(f"Training epoch {str(epoch)}")
        total_reward = 0.0
        reward_vals = []
    #note: if buggy, try parentheses
        for state_agent_features1, state_agent_features2, action1, action2, reward in train_data_loader:
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # Move to device
            state_agent_features1, state_agent_features2 = state_agent_features1.to(device), state_agent_features2.to(device)
            reward = reward.to(device)
            action1, action2 = action1.to(device), action2.to(device)
            #expert_action1, expert_action2 = expert_action1.to(device), expert_action2.to(device)
      

            out_logits1 = model_player_1(state_agent_features1) #B:6
            out_logits2 = model_player_2(state_agent_features2) #B:6
            #print(out_logits2.shape)
            # colon because batched?
            pi_accel_1 = Normal(out_logits1[:,0], torch.nn.functional.softplus(out_logits1[:,1]) + 1e-4) #BNorms
            pi_accel_2 = Normal(out_logits2[:,0], torch.nn.functional.softplus(out_logits2[:,1]) + 1e-4)
            pi_steer_1 = Normal(out_logits1[:,2], torch.nn.functional.softplus(out_logits1[:,3]) + 1e-4)
            pi_steer_2 = Normal(out_logits2[:,2], torch.nn.functional.softplus(out_logits2[:,3]) + 1e-4)
            pi_brake_1 = Bernoulli(logits=out_logits1[:,4]) #BBern
            pi_brake_2 = Bernoulli(logits=out_logits2[:,4])
            ###TODO: Drift?
            pi_drift_1 = Bernoulli(logits=out_logits1[:, 5])  # BBern
            pi_drift_2 = Bernoulli(logits=out_logits2[:, 5])
            #action1 should be B:1, and 6 total actions
            expected_log_accel_1 = (pi_accel_1.log_prob(action1[:,0])*reward).mean()
            expected_log_accel_2 = (pi_accel_2.log_prob(action2[:,0])*reward).mean()
            expected_log_steer_1 = (pi_steer_1.log_prob(action1[:,1])*reward).mean()
            expected_log_steer_2 = (pi_steer_2.log_prob(action2[:,1])*reward).mean()
            expected_log_brake_1 = (pi_brake_1.log_prob(action1[:,2])*reward).mean()
            expected_log_brake_2 = (pi_brake_2.log_prob(action2[:,2])*reward).mean()
            
            expected_log_full = (expected_log_accel_1 + expected_log_accel_2 + expected_log_steer_1 + expected_log_steer_2 + expected_log_brake_1 + expected_log_brake_2)/6
             
            (-expected_log_full).backward()

            reward_vals.append(expected_log_full.detach().cpu().numpy())
            total_reward += expected_log_full.detach().cpu().numpy()
            optimizer_1.step()
            optimizer_2.step()

        total_avg_reward = total_reward/len(reward_vals)
        print(f"Average reward per frame for epoch {epoch}: {total_avg_reward}")
        print(f'\treward 1: {sum(reward_vals) / len(reward_vals)}')
        if (total_avg_reward > best_total_avg_reward):
            print(f'\tSaving new best model at epoch {epoch}')
            save_model(model_player_1, f"reinforce_best_1")
            save_model(model_player_2, f"reinforce_best_2")
            save_model(model_player_1, f"reinforce_best_1_{outer_epoch}")
            save_model(model_player_2, f"reinforce_best_2_{outer_epoch}")
            best_total_avg_reward = total_avg_reward
        save_model(model_player_1, f"reinforce_1_{outer_epoch}_{epoch}")
        save_model(model_player_2, f"reinforce_2_{outer_epoch}_{epoch}")

def train_dagger_outer(args):
    # Train initial loop with given models
    print(f'Starting outer epoch {args.outer_epoch}...')
    train_dagger(args, initial=True, outer_epoch=args.outer_epoch)
    #max_epochs = 10
    #for epoch in range(1, max_epochs):
    #    print(f'Starting outer epoch {epoch}...')
    #    train_dagger(args, initial=False, outer_epoch=epoch)

def train_reinforce_outer(args):
    # Train initial loop with given models
    print('Starting outer epoch 0...')

    train_reinforce(args, initial=True, outer_epoch=0)
    max_epochs = 10
    for epoch in range(1, max_epochs):
        print(f'Starting outer epoch {epoch}...')
        train_reinforce(args, initial=False, outer_epoch=epoch)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('training', help='\'imitation\' or \'dagger\' training')
    parser.add_argument('--log-dir')
    parser.add_argument('--model1', help='model1 name for dagger training')
    parser.add_argument('--model2', help='model2 name for dagger training')
    # Put custom arguments here
    parser.add_argument('-e', '--epoch', type=int, default=10)
    # Janky way to label epochs
    parser.add_argument('-o', '--outer_epoch', type=int)

    args = parser.parse_args()
    if args.training == 'imitation':
        train_imitation(args)
    elif args.training == 'dagger':
        train_dagger_outer(args)
    elif args.training == 'reinforce':
        train_reinforce_outer(args)