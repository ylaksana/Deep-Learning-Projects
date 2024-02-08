import torch
import numpy as np
from state_agent.utils import load_imitation_data, load_imitation_sequence_data
from state_agent.state_agent_model import Imitator, SequenceImitator, save_model
import torch.utils.tensorboard as tb
from datetime import datetime
import torch.optim as optim
import torch.nn as nn



def imitation_loss(logits_sequence, expert_actions_sequence):
    # Get the sequence length
    seq_len = logits_sequence.size(1)

    # Initialize total loss
    total_loss = 0.0

    for t in range(seq_len):
        # Extract logits and expert actions for the current time step
        logits_t = logits_sequence[:, t]
        expert_actions_t = expert_actions_sequence[:, t]

        # Extract continuous and discrete parts from logits
        mean_acc = logits_t[:, 0]
        var_acc = logits_t[:, 1]
        mean_steer = logits_t[:, 2]
        var_steer = logits_t[:, 3]
        discrete_logits = logits_t[:, 4:6]

        # Ensure variances are positive (e.g., by using softplus)
        var_acc = torch.nn.functional.softplus(var_acc)
        var_steer = torch.nn.functional.softplus(var_steer)

        # Continuous expert actions
        continuous_expert_actions = expert_actions_t[:, :2]

        # Discrete expert actions
        discrete_expert_actions = expert_actions_t[:, 2:]

        # Loss for continuous actions at the current time step
        gaussian_nll_loss = nn.GaussianNLLLoss()
        loss_continuous_acc = gaussian_nll_loss(mean_acc, continuous_expert_actions[:, 0], var_acc)
        loss_continuous_steer = gaussian_nll_loss(mean_steer, continuous_expert_actions[:, 1], var_steer)
        loss_continuous = loss_continuous_acc + loss_continuous_steer

        # Loss for discrete actions at the current time step
        bce_with_logits_loss = nn.BCEWithLogitsLoss()
        loss_discrete = bce_with_logits_loss(discrete_logits, discrete_expert_actions)

        # Add the losses at the current time step to the total loss
        total_loss += (loss_continuous + loss_discrete)
    total_loss /= seq_len
    return total_loss
def train(args):
    from os import path
    model = SequenceImitator(input_size=52)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{model}")
    if device is not None:
        print(f"moving model to {device}")
        model = model.to(device)
    train_logger, valid_logger = None, None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, f'valid{current_time}'), flush_secs=1)

    max_epochs = 20
    save_freq = 10
    scheduler_enabled = True
    print("Started Loading data")
    train_dataset_loader = load_imitation_sequence_data(
        '/Users/michaelhuang/Library/CloudStorage/GoogleDrive-mghuang21@gmail.com/My Drive/MSCS/DLFALL2023/SuperTuxHocky/imitation_data', max_pkls=100, window_size=30, window_stride=2)
    print("Finished loading data")

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    criterion = imitation_loss

    dataset_size = len(train_dataset_loader)
    print(f"Starting training loop for dataset_size: {dataset_size}")
    train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], 0)
    for epoch in range(0, max_epochs):
        model.train()
        print(f"Training epoch {str(epoch)}")
        i = 0
        for batch_x, batch_y in train_dataset_loader:
            if device is not None:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            #print(f"inputs: {batch_x.shape} labels: {batch_y.shape}")

            # Forward pass
            outputs = model(batch_x, batched=True)
            # print("forward pass finished")
            #print(f"outputs: {outputs.shape} ")
            loss = criterion(outputs, batch_y)

            train_logger.add_scalar('loss', float(loss), epoch * dataset_size + i)
            loss.backward()
            optimizer.step()
            i += 1

        if scheduler_enabled:
            scheduler.step()  # val_acc)
            train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch * dataset_size + i - 1)
        if epoch % save_freq == 0:
            save_model(model, f"{epoch}")
    save_model(model, "final")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log-dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
