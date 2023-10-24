from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import accuracy, load_data
import torch.utils.tensorboard as tb
import numpy as np

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))
    """
    Your code here, modify your HW1 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    # Initialize hyperparameters
    num_epochs = 50

    # Configure device
    model.to(device)

    # Create Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # Create loss function
    loss = ClassificationLoss()
    
    # Load dataset
    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    # load_model()
    
    # Start training
    global_step = 0
    for epoch in range(num_epochs):
        # Create lists for accuracy and validation accuracy
        model.train()
        loss_vals,acc_vals, vacc_vals = [], [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # Model the data
            logit = model(img)

            # Calculate loss
            loss_val = loss(logit, label.long())

            # Calculate accuracy
            acc_val = accuracy(logit, label.long())
            print(acc_val.shape)

            # Append loss and accuracy calculations to lists
            loss_vals.append(loss_val)
            # acc_vals.append(acc_val.detach().cpu().numpy())

            # Log the loss avg
            train_logger.add_scalar('loss', float(loss_val), global_step=global_step)
            
            #Calculate gradient
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Update global step
            global_step += 1

        avg_loss = sum(loss_vals) / len(loss_vals)
        # avg_acc = np.mean(acc_vals)
        # train_logger.add_scalar('accuracy', avg_acc, global_step = global_step)

        model.eval()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            vacc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        # avg_vacc = np.mean(vacc_vals)
        # valid_logger.add_scalar('accuracy', avg_vacc, global_step = global_step)

        print('epoch %-3d \t loss = %0.4f' % (epoch, avg_loss))
    
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['cnn'])
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-log_dir','--log_dir',type=str, default ='data')

    args = parser.parse_args()
    train(args)

