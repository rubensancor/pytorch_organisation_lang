import argparse
import time

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import MultiCNN
from load_data import load_dataset
from metrics import f1, prec, rec
from pytorchtools import EarlyStopping
import wandb

wandb.init(project="organisational-language")

parser = argparse.ArgumentParser(description='Main file')
# parser.add_argument('-f', '--filename',
#                     help='File with the configuration for the experiment')
# args = parser.parse_args()
# with open(args.filename) as file:
#     print('File')
#     # TODO Manage the data inserted in the experiment
args = parser.parse_args()

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def initializate_model():
    model = MultiCNN(dropout_prob=0.8,
                     embedding_matrix=word_embeddings,
                     vocab_size=vocab_size,
                     embedding_length=300,
                     kernel_heights=[3, 4, 5, 6],
                     in_channels=1,
                     out_channels=200,
                     num_labels=6)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer


def train_model(model, optim, train_iter, epoch):
    t = time.time()
    total_epoch_loss = []
    total_epoch_prec = []
    total_epoch_f1 = []
    total_epoch_rec = []

    # Train
    model.cuda()
    model.train()
    for idx, batch in enumerate(train_iter):

        # Get inputs from batch
        text = batch.tweet[0]
        target = batch.organisation
        target = torch.autograd.Variable(target).long()

        # Convert inputs to GPU
        # text = text.cuda()
        # target = target.cuda()

        # Set the parameter gradients to zero
        optim.zero_grad()

        # Make the predictions for the text
        prediction = model(text)
        
        # Calculate the metrics
        loss = F.cross_entropy(prediction, target)
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
                  'f1_train: {:.4f}'.format(np.mean(total_epoch_f1)),
                  'prec_train: {:.4f}'.format(np.mean(total_epoch_prec)),
                  'rec_train: {:.4f}'.format(np.mean(total_epoch_rec)),
                  'loss_train_moment: {:.4f}'.format(loss.item()),
                  'loss_train_mean: {:.4f}'.format(np.mean(total_epoch_loss)),
                  'time: {:.4f}s'.format(time.time() - t),
                  flush=True)

    # Log train metrics to wandb
    wandb.log({'epoch': epoch+1,
                'f1_train': np.mean(total_epoch_f1),
                'prec_train': np.mean(total_epoch_prec),
                'rec_train': np.mean(total_epoch_rec),
                'loss_train_moment': loss.item(),
                'loss_train_mean': np.mean(total_epoch_loss)})

    return total_epoch_loss, total_epoch_f1, total_epoch_prec, total_epoch_rec


def val_model(model, val_iter, epoch, patience):
    total_f1 = []
    total_prec = []
    total_rec = []
    total_loss = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)


    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            
            text = batch.tweet[0]
            target = batch.organisation
            target = torch.autograd.Variable(target).long()

            prediction = model(text)

            loss = F.cross_entropy(prediction, target)
            total_f1.append(f1(prediction, target))
            total_prec.append(prec(prediction, target))
            total_rec.append(rec(prediction, target))
            total_loss.append(loss.item())

    print('VALIDATION --> '
          'f1_val: {:.4f}'.format(np.mean(total_f1)),
          'prec_val: {:.4f}'.format(np.mean(total_prec)),
          'rec_val: {:.4f}'.format(np.mean(total_rec)),
          'loss_val: {:.4f}'.format(loss.item()),
          flush=True)
    wandb.log({'epoch': epoch+1,
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

    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            text = batch.tweet[0]
            target = batch.organisation
            target = torch.autograd.Variable(target).long()

            prediction = model(text)
            loss = F.cross_entropy(prediction, target)
            total_f1_test.append(f1(prediction, target))
            total_prec_test.append(prec(prediction, target))
            total_rec_test.append(rec(prediction, target))
            total_loss_test.append(loss.item())

    print('TEST --> ',
          'f1_test: {:.4f}'.format(np.mean(total_f1_test)),
          'prec_test: {:.4f}'.format(np.mean(total_prec_test)),
          'rec_test: {:.4f}'.format(np.mean(total_rec_test)),
          'loss_test: {:.4f}'.format(loss.item()),
          flush=True)
    wandb.log({'f1_test': np.mean(total_f1_test),
               'prec_test': np.mean(total_prec_test),
               'rec_test': np.mean(total_rec_test),
               'loss_test_mean': np.mean(total_loss_test)})
    


if __name__ == '__main__':
    print('*' * 20 + ' Loading data ' + '*' * 20, flush=True)

    (TEXT,
     vocab_size,
     word_embeddings,
     train_iter,
     valid_iter,
     test_iter) = load_dataset('./data/mixed_nothing.csv', device)

    print('*' * 20 + ' DATA LOADED! ' + '*' * 20, flush=True)

    model, optimizer = initializate_model()
    wandb.watch(model, log="all")
    
    par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has %i trainable parameters' % par, flush=True)

    EPOCH = 10
    for i in range(EPOCH):
        train_model(model, optimizer, train_iter, i)
        early_stop = val_model(model, valid_iter, i, 3)

        if early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    test_model(model, test_iter)
