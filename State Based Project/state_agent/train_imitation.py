import torch
import numpy as np
from state_agent.utils import load_imitation_data, load_imitation_jurgen_data
from state_agent.state_agent_model import Imitator, save_model, load_model
import torch.utils.tensorboard as tb
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

'''
def imitation_loss(logits, expert_actions):
    # Split logits and expert actions into two halves
    logits_first_half = logits[:, :6]
    logits_second_half = logits[:, 6:]
    expert_actions_first_half = expert_actions[:, :4]
    expert_actions_second_half = expert_actions[:, 4:]

    # Function to calculate loss for each half
    def calculate_half_loss(logits_half, expert_actions_half):
        # Extract continuous and discrete parts from logits
        mean_acc = logits_half[:, 0]
        var_acc = logits_half[:, 1]
        mean_steer = logits_half[:, 2]
        var_steer = logits_half[:, 3]
        discrete_logits = logits_half[:, 4:6]

        # Ensure variances are positive
        var_acc = F.softplus(var_acc)
        var_steer = F.softplus(var_steer)

        # Continuous expert actions
        continuous_expert_actions = expert_actions_half[:, :2]

        # Discrete expert actions
        discrete_expert_actions = expert_actions_half[:, 2:]

        # Loss for continuous actions
        gaussian_nll_loss = nn.GaussianNLLLoss()
        loss_continuous_acc = gaussian_nll_loss(mean_acc, continuous_expert_actions[:, 0], var_acc)
        loss_continuous_steer = gaussian_nll_loss(mean_steer, continuous_expert_actions[:, 1], var_steer)
        loss_continuous = loss_continuous_acc + loss_continuous_steer

        # Loss for discrete actions
        bce_with_logits_loss = nn.BCEWithLogitsLoss()
        loss_discrete = bce_with_logits_loss(discrete_logits, discrete_expert_actions)

        return loss_continuous + loss_discrete

    # Calculate loss for each half and sum them
    total_loss = calculate_half_loss(logits_first_half, expert_actions_first_half) + \
                 calculate_half_loss(logits_second_half, expert_actions_second_half)

    return total_loss
'''
def imitation_loss(logits, expert_actions):
    # Split logits and expert actions into two halves
    logits_first_half = logits[:, :4]
    logits_second_half = logits[:, 4:]
    expert_actions_first_half = expert_actions[:, :4]
    expert_actions_second_half = expert_actions[:, 4:]

    # Function to calculate loss for each half
    def calculate_half_loss(logits_half, expert_actions_half):
        # Extract continuous and discrete parts from logits
        acc = logits_half[:,0]
        steer = logits_half[:,1]
        brake = logits_half[:,2]
        drift = logits_half[:,3]

        # Loss for continuous actions
        mse_loss = nn.MSELoss()
        loss_continuous_steer = mse_loss(steer, expert_actions_half[:, 1])

        # Loss for discrete actions
        bce_with_logits_loss = nn.BCEWithLogitsLoss()
        loss_discrete_acc = bce_with_logits_loss(acc, expert_actions_half[:, 0])
        loss_discrete_brake = bce_with_logits_loss(brake, expert_actions_half[:, 2])
        loss_discrete_drift = bce_with_logits_loss(drift, expert_actions_half[:, 3])
        loss_discrete = loss_discrete_acc + loss_discrete_brake + loss_discrete_drift

        return loss_continuous_steer + loss_discrete

    # Calculate loss for each half and sum them
    total_loss = calculate_half_loss(logits_first_half, expert_actions_first_half) + \
                 calculate_half_loss(logits_second_half, expert_actions_second_half)

    return total_loss

def train_imitation(args):
    from os import path
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model =  Imitator(input_size=34)
    print(f"{model}")
    print(f"moving model to {device}")
    model = model.to(device)
    #model = load_model('imitation_jurgen_final').to(device)

    train_logger = None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)

    scheduler_enabled = True
    print("Started Loading data")
    #train_dataset_loader = load_imitation_data('imitation_data', max_pkls=100)
    train_dataset_loader = load_imitation_jurgen_data('imitation_data_jurgen_agent_large', max_pkls=150)
    print("Finished loading data")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=0.0001)
    criterion = imitation_loss

    dataset_size = len(train_dataset_loader.dataset)
    print(f"Starting training loop for dataset_size: {dataset_size}")
    if scheduler_enabled and train_logger is not None:
        train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], 0)

    start_epoch = 0
    max_epochs = args.epoch + start_epoch
    for epoch in range(start_epoch, max_epochs):
        model.train()
        model.to(device)
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
            loss_vals.append(loss)

            if train_logger is not None:
                train_logger.add_scalar('loss', float(loss), epoch * dataset_size + i)
            loss.backward()
            optimizer.step()
            i += 1
        avg_loss = sum(loss_vals) / len(loss_vals) 
        print(f'\tloss = {avg_loss}')

        if scheduler_enabled:
            scheduler.step(avg_loss)
            if train_logger is not None:
                train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch * dataset_size + i - 1)
        save_model(model, f"imitation_jurgen_{epoch}")
    save_model(model, f"imitation_jurgen_{epoch}")
    save_model(model, "imitation_jurgen_final")
    # Inverse Reinforce Learning, Inverse Q Learning. Try to imitate a hard coded agent

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir')
    parser.add_argument('-e', '--epoch', type=int, default=20)
    args = parser.parse_args()
    train_imitation(args)