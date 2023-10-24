import torch
import numpy as np

from .models import FCN, save_model, ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, invert_dense_distribution
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Configure device
    model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Create Step Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # Create transforms.compose for data augmentation
    augment = dense_transforms.Compose([
            dense_transforms.ColorJitter(saturation=0.5, hue = 0.5, contrast = 0.3),
            dense_transforms.RandomHorizontalFlip(),
            dense_transforms.ToTensor()
        ])

    # Load the dataset
    train_data = load_dense_data('dense_data/train', augment)
    valid_data = load_dense_data('dense_data/valid')

    # Set global step
    global_step = 0

    # Invert, normalize and convert DENSE_CLASS_DISTRIBUTION into weight tensor:
    inverted_distribution = invert_dense_distribution(DENSE_CLASS_DISTRIBUTION)
    weight_tensor = torch.tensor(inverted_distribution)
    weight_tensor = weight_tensor.to(device)

    loss = torch.nn.CrossEntropyLoss(weight = weight_tensor)
    # Start Training
    for epoch in range(10):
        model.train()
        # Initialize confusion matrix
        acc = ConfusionMatrix()
        loss_vals= []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            
            # Insert images into the model
            logit = model(img)

            # Calculate loss
            loss_val = loss(input = logit, target = label.long())

            # Append loss
            loss_vals.append(loss_val)

            # Update confusion matrix
            acc.add(logit.argmax(1), label)

            #Log loss calculation
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)


            # Calculate gradient
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Update global step
            global_step += 1
        
        # Average the accuracy of epoch
        avg_acc = acc.global_accuracy


        # Average loss of epoch
        avg_loss = sum(loss_vals) / len(loss_vals)

        # Log average
        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        val_matrix = ConfusionMatrix()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            # Calculate and append validation accuracy
            logit = model(img)
            val_matrix.add(logit.argmax(1), label)
        # Calculate the average validation accuracy
        avg_vacc = val_matrix.global_accuracy

        # Append validation accuracy
        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f \t IoU = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc, acc.iou))
        

        
        scheduler.step(avg_vacc)
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-b','--batch_size', type=int, default=128)
    parser.add_argument('--no_nomrlaization', action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-log_dir','--log_dir',type=str, default ='data')
    args = parser.parse_args()
    train(args)
