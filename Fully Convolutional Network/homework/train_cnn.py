from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
from torchvision import transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Configure device
    model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Create Step Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # Create loss function
    loss = torch.nn.CrossEntropyLoss()

    # Create transforms.compose for data augmentation
    augment = transforms.Compose([
            transforms.ColorJitter(saturation=0.5),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor()
        ])

    # Load the dataset
    train_data = load_data('data/train', augment)
    valid_data = load_data('data/valid')

    # Set global step
    global_step = 0


    # Start Training
    for epoch in range(args.num_epoch):
        model.train()
        acc_vals, loss_vals = [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            
            logit = model((img))

            # Calculate loss
            loss_val = loss(logit, label)

            # Append loss
            loss_vals.append(loss_val)

            # Calculate accuracy
            acc_val = accuracy(logit, label)

            #Log loss calculation
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            # Append accuracy
            acc_vals.append(acc_val.detach().cpu().numpy())

            # Calculate gradient
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Update global step
            global_step += 1
        
        # Average the accuracy of epoch
        avg_acc = torch.mean(acc_val)

        # Average loss of epoch
        avg_loss = sum(loss_vals) / len(loss_vals)

        # Log average
        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            # Calculate and append accuracy on validation set
            acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        # Calculate the average validation accuracy
        avg_vacc = sum(acc_vals) / len(acc_vals)
        vacc = sum(avg_vacc) / len(avg_vacc)

        # Append validation accuracy
        if valid_logger:
            valid_logger.add_scalar('accuracy', vacc, global_step)

        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, vacc))
        
        # if(vacc > 0.9):
        #     break
        
        scheduler.step(vacc)
        
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['cnn'])
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-b','--batch_size', type=int, default=128)
    parser.add_argument('--no_nomrlaization', action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-log_dir','--log_dir',type=str, default ='data')
    args = parser.parse_args()
    train(args)
