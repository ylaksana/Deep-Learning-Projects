import torch
import numpy as np

import geoffrey_agent
import image_jurgen_agent
import jurgen_agent
import yann_agent
from state_agent_dagger.utils import load_imitation_data, load_dagger_data, extract_features
from state_agent_dagger.state_agent_model import Imitator, save_model, load_model
import torch.utils.tensorboard as tb
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import tournament.runner as runner
from tournament.utils import StateRecorder, load_recording
from geoffrey_agent import Team


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
    if device is not None:
        print(f"moving model to {device}")
        model = model.to(device)

    train_logger = None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)

    max_epochs = 5
    scheduler_enabled = True
    print("Started Loading data")
    train_dataset_loader = load_imitation_data('imitation_data', max_pkls=50)
    print("Finished loading data")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    criterion = imitation_loss

    dataset_size = len(train_dataset_loader)
    print(f"Starting training loop for dataset_size: {dataset_size}")
    if scheduler_enabled and train_logger is not None:
        train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], 0)

    for epoch in range(0, max_epochs):
        model.train()
        print(f"Training epoch {str(epoch)}")
        i = 0
        for batch_x, batch_y in train_dataset_loader:
            if device is not None:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logit = model.forward(batch_x)
            # print("forward pass finished")
            #print(logit.shape)
            #print(batch_y.shape)
            loss = criterion(logit, batch_y)

            if train_logger is not None:
                train_logger.add_scalar('loss', float(loss), epoch * dataset_size + i)
            loss.backward()
            optimizer.step()
            i += 1

        if scheduler_enabled and train_logger is not None:
            scheduler.step()  # val_acc)
            train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch * dataset_size + i - 1)
        save_model(model, f"imitation_{epoch}")
    save_model(model, "imitation_final")

def run_match(state_file, team_1, team_2):
    import os
    record_state = state_file
    num_players = 2
    num_frames = 1200
    max_score = 3
    result = None
    team1 = team_1
    team2 = team_2
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
        result = match.run(team1, team2, num_players, num_frames, max_score=max_score, record_fn=recorder)
    except runner.MatchException as e:
        print('Match failed', e.score)
        print(' T1:', e.msg1)
        print(' T2:', e.msg2)

    rollout = load_recording(record_state)
    states = []
    for frame in rollout:
        states.append(frame)

    #os.remove(record_state)
    return states, result


def train_dagger(args):
    from os import path, remove
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model_player_1 = load_model(args.model1)
    model_player_2 = load_model(args.model2)

    if device is not None:
        print(f"moving model to {device}")
        model_player_1, model_player_2 = model_player_1.to(device), model_player_2.to(device)
    train_logger = None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)
        print(f'Train logger successfully loaded.')

    max_epochs = 50
    scheduler_enabled = False
    frames_batch_size = 1028

    optimizer_1, optimizer_2 = optim.Adam(model_player_1.parameters(), lr=0.001, weight_decay=0.001), optim.Adam(model_player_2.parameters(), lr=0.001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    criterion = imitation_loss
    global_step = 0

    print(f"Starting training loop ")
    if scheduler_enabled and train_logger is not None:
        train_logger.add_scalar('LR', optimizer_1.param_groups[0]['lr'], 0)
    opponent_agent_list = ["yoshua_agent", "yann_agent", "jurgen_agent", "geoffrey_agent", "image_jurgen_agent"]
    expert_agent = geoffrey_agent.Team()
    expert_agent.new_match(0, 2)
    for epoch in range(0, max_epochs):
        model_player_1.train()
        model_player_2.train()
        print(f"Training epoch {str(epoch)}")
        total_loss = 0.0
        total_frames = 0
        for opponent in opponent_agent_list:
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            print(f'match in progress: state_agent vs {opponent}')
            trajectory, result = run_match(f"dagger_data/temp_run.pkl", "state_agent_dagger", opponent)
            print(f'match result: {result} number of frames {len(trajectory)}')
            for frame in trajectory:
                total_frames += 1
                state1, state2 = (frame['team1_state'], frame['team2_state'], frame['soccer_state']),  ((frame['team1_state'][1],frame['team1_state'][0]), frame['team2_state'], frame['soccer_state'])
                # ask the expert
                expert_actions = expert_agent.act(frame['team1_state'], frame['team2_state'], frame['soccer_state'])
                expert_action_tensors = []
                for expert_act_dict in expert_actions:
                    drift = 0
                    if 'drift' in expert_act_dict:
                        drift = expert_act_dict['drift'][0]
                    actions_values = [expert_act_dict['acceleration'][0], expert_act_dict['steer'][0],
                                      expert_act_dict['brake'][0],
                                      drift]
                    expert_act_tensor = torch.tensor(actions_values, dtype=torch.float32)
                    expert_action_tensors.append(expert_act_tensor)

                # ask our models
                state_agent_features_1, state_agent_features_2 = extract_features(state1, 0), extract_features(state2, 0)
                if device != None:
                    state_agent_features_1, state_agent_features_2 = state_agent_features_1.to(device), state_agent_features_2.to(device)
                    expert_action_tensors[0], expert_action_tensors[1] = expert_action_tensors[0].to(device), expert_action_tensors[1].to(device)


                out_logits_1, out_logits_2 = model_player_1(state_agent_features_1), model_player_2(state_agent_features_2)
                loss1, loss2 = criterion(out_logits_1.unsqueeze(0), expert_action_tensors[0].unsqueeze(0)), criterion(out_logits_2.unsqueeze(0), expert_action_tensors[1].unsqueeze(0))
                loss1.backward()
                loss2.backward()
                total_loss += (loss1 + loss2)
                optimizer_1.step()
                optimizer_2.step()
            # Remove temp file at end to avoid race condition
            remove(f"dagger_data/temp_run.pkl")

        print(f"Average loss per frame for epoch {epoch}: {total_loss}")
        train_logger.add_scalar("Average loss per epoch", total_loss, global_step=global_step)
        global_step+=1
        save_model(model_player_1, f"dagger_1_{epoch}")
        save_model(model_player_2, f"dagger_2_{epoch}")
    save_model(model_player_1, f"dagger_final_1")
    save_model(model_player_2, f"dagger_final_2")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-log_dir','--log-dir',type=str,default='data')
    parser.add_argument('--model1')
    parser.add_argument('--model2')
    # Put custom arguments here

    args = parser.parse_args()
    #train_imitation(args)
    train_dagger(args)