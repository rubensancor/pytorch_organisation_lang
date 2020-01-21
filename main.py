import os
import time
import wandb
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import MultiCNN
from load_data import load_dataset
from metrics import f1, prec, rec, acc
from pytorchtools import EarlyStopping

parser = argparse.ArgumentParser(description='Main file')
parser.add_argument('-ws', '--wandb_sync',
                    action='store_true',
                    help='Check if the run is going to be uploaded to WandB')
parser.add_argument('--freeze',
                    action='store_true',
                    help='Freeze pretrained embeddings.')
parser.add_argument('-f', '--file',
                    required=True,
                    help='The dataset file to use. It must be stored in '
                         './data folder')
parser.add_argument('-p', '--patience',
                    default=2, type=int,
                    help=('The number of epochs that have to pass without '
                          'reducing the val_loss prior to call early '
                          'stopping'))
parser.add_argument('-e', '--epochs',
                    default=10, type=int,
                    help=('Number of epochs to run for the experiment.'))
parser.add_argument('-b', '--batch',
                    default=1024, type=int,
                    help=('Batch size for each step.'))
parser.add_argument('-d', '--dropout',
                    default=0.8, type=float,
                    help=('Dropout probability for the model.'))
parser.add_argument('--lr',
                    default=0.001, type=float,
                    help=('Learning rate for the optimizer.'))
parser.add_argument('--seed',
                    default=1234, type=int,
                    help=('Seed for the random generator.'))
args = parser.parse_args()

if not args.wandb_sync:
    os.environ['WANDB_MODE'] = 'dryrun'

wandb.init(project="organisational-language", config=args)

seed = args.seed

np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def initializate_model(dropout, word_embeddings, vocab_size,
                       num_labels, embedding_length, kernel_heights,
                       in_channels, out_channels, mixed_memory,
                       freeze_embeddings, lr):

    model = MultiCNN(dropout_prob=dropout,
                     embedding_matrix=word_embeddings,
                     vocab_size=vocab_size,
                     embedding_length=embedding_length,
                     kernel_heights=kernel_heights,
                     in_channels=in_channels,
                     out_channels=out_channels,
                     mixed_memory=mixed_memory,
                     num_labels=num_labels,
                     freeze_embeddings=freeze_embeddings)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


def train_model(model, optim, train_iter, epoch):
    t = time.time()
    total_epoch_loss = []
    total_epoch_prec = []
    total_epoch_f1 = []
    total_epoch_rec = []
    total_epoch_acc = []

    # Train
    if not model.mixed_memory:
        model = model.to(device)

    model.train()
    for idx, batch in enumerate(train_iter):

        # Get inputs from batch
        text = batch.tweet[0]
        target = batch.organisation
        target = torch.autograd.Variable(target).long()

        # Convert inputs to GPU
        # text = text.to(device)
        target = target.to(device)

        # Set the parameter gradients to zero
        optim.zero_grad()

        # Make the predictions for the text
        prediction = model(text)

        # Calculate the metrics
        loss = F.cross_entropy(prediction, target)
        total_epoch_acc.append(acc(prediction, target))
        total_epoch_f1.append(f1(prediction, target))
        total_epoch_prec.append(prec(prediction, target))
        total_epoch_rec.append(rec(prediction, target))
        total_epoch_loss.append(loss.item())
        #
        loss.backward()
        optim.step()

        if idx % 100 == 0:
            print('TRAIN --> ',
                  'Epoch: {:04d}'.format(epoch + 1),
                  'acc_train: {:04f}'.format(np.mean(total_epoch_acc)),
                  'f1_train: {:.4f}'.format(np.mean(total_epoch_f1)),
                  'prec_train: {:.4f}'.format(np.mean(total_epoch_prec)),
                  'rec_train: {:.4f}'.format(np.mean(total_epoch_rec)),
                  'loss_train_moment: {:.4f}'.format(loss.item()),
                  'loss_train_mean: {:.4f}'.format(np.mean(total_epoch_loss)),
                  'time: {:.4f}s'.format(time.time() - t),
                  flush=True)

    # Log train metrics to wandb
    wandb.log({'epoch': epoch+1,
               'acc_train': np.mean(total_epoch_acc),
               'f1_train': np.mean(total_epoch_f1),
               'prec_train': np.mean(total_epoch_prec),
               'rec_train': np.mean(total_epoch_rec),
               'loss_train_moment': loss.item(),
               'loss_train_mean': np.mean(total_epoch_loss)})

    return total_epoch_loss, total_epoch_f1, total_epoch_prec, total_epoch_rec


def val_model(model, val_iter, epoch, early_stopping):
    total_f1 = []
    total_prec = []
    total_rec = []
    total_loss = []
    total_acc = []

    if not model.mixed_memory:
        model = model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(val_iter):

            text = batch.tweet[0]
            target = batch.organisation
            target = torch.autograd.Variable(target).long()

            # Convert inputs to GPU
            # text = text.to(device)
            target = target.to(device)

            prediction = model(text)

            loss = F.cross_entropy(prediction, target)
            total_acc.append(acc(prediction, target))
            total_f1.append(f1(prediction, target))
            total_prec.append(prec(prediction, target))
            total_rec.append(rec(prediction, target))
            total_loss.append(loss.item())

    print('VALIDATION --> '
          'acc_val: {:.4f}'.format(np.mean(total_acc)),
          'f1_val: {:.4f}'.format(np.mean(total_f1)),
          'prec_val: {:.4f}'.format(np.mean(total_prec)),
          'rec_val: {:.4f}'.format(np.mean(total_rec)),
          'loss_val: {:.4f}'.format(loss.item()),
          flush=True)
    wandb.log({'epoch': epoch+1,
               'acc_val': np.mean(total_acc),
               'f1_val': np.mean(total_f1),
               'prec_val': np.mean(total_prec),
               'rec_val': np.mean(total_rec),
               'loss_val_mean': np.mean(total_loss)})

    early_stopping(np.mean(total_loss), model)

    return early_stopping.early_stop


def test_model(model, test_iter):
    total_f1_test = []
    total_prec_test = []
    total_rec_test = []
    total_loss_test = []
    total_acc_test = []

    if not model.mixed_memory:
        model = model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            text = batch.tweet[0]
            target = batch.organisation
            target = torch.autograd.Variable(target).long()

            # Convert inputs to GPU
            # text = text.to(device)
            target = target.to(device)

            prediction = model(text)

            loss = F.cross_entropy(prediction, target)
            total_acc_test.append(acc(prediction, target))
            total_f1_test.append(f1(prediction, target))
            total_prec_test.append(prec(prediction, target))
            total_rec_test.append(rec(prediction, target))
            total_loss_test.append(loss.item())

    print('TEST --> ',
          'acc_test: {:.4f}'.format(np.mean(total_acc_test)),
          'f1_test: {:.4f}'.format(np.mean(total_f1_test)),
          'prec_test: {:.4f}'.format(np.mean(total_prec_test)),
          'rec_test: {:.4f}'.format(np.mean(total_rec_test)),
          'loss_test: {:.4f}'.format(loss.item()),
          flush=True)
    wandb.log({'acc_test': np.mean(total_acc_test),
               'f1_test': np.mean(total_f1_test),
               'prec_test': np.mean(total_prec_test),
               'rec_test': np.mean(total_rec_test),
               'loss_test_mean': np.mean(total_loss_test)})


if __name__ == '__main__':
    print('*' * 20 + ' Loading data ' + '*' * 20, flush=True)

    # TODO Change the device to use with a flag
    (TEXT,
     vocab_size,
     label_size,
     word_embeddings,
     train_iter,
     valid_iter,
     test_iter) = load_dataset(path='./data/' + args.file,
                               device=torch.device('cpu'),
                               batch_size=args.batch)

    print('*' * 20 + ' DATA LOADED! ' + '*' * 20, flush=True)

    model, optimizer = initializate_model(dropout=args.dropout,
                                          word_embeddings=word_embeddings,
                                          vocab_size=vocab_size,
                                          num_labels=label_size,
                                          embedding_length=300,
                                          kernel_heights=[2, 3, 4, 5],
                                          in_channels=1,
                                          out_channels=200,
                                          mixed_memory=True,
                                          freeze_embeddings=False,
                                          lr=args.lr)
    wandb.watch(model, log="all")

    par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has %i trainable parameters' % par, flush=True)

    EPOCHS = args.epochs
    patience = args.patience
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for i in range(EPOCHS):
        train_model(model, optimizer, train_iter, i)
        early_stop = val_model(model, valid_iter, i, early_stopping)

        if early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    test_model(model, test_iter)
