import argparse
import time

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import MultiCNN
from load_data import load_dataset
import wandb

wandb.init(project="organisational-language")

# parser = argparse.ArgumentParser(description='Main file')
# parser.add_argument('-f', '--filename',
#                     help='File with the configuration for the experiment')
# args = parser.parse_args()
# with open(args.filename) as file:
#     print('File')
#     # TODO Manage the data inserted in the experiment

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset('Organisation_lang/pytorch/mixed.csv')


def initializate_model():
    model = MultiCNN(0.8, word_embeddings, vocab_size, 300, [2, 3, 4, 5], 1, 200)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer


def train_model(model, optim, train_iter, epoch):
    t = time.time()
    total_epoch_loss = 0
    total_epoch_acc = 0
    
    # Train
    model.train()
    model.double()
    steps = 0
    for idx, batch in enumerate(train_iter):
        text = batch.tweet[0]
        target = batch.organisation
        target = torch.autograd.Variable(target).long()

        optim.zero_grad()
        prediction = model(text)
        loss = F.cross_entropy(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        optim.step()
        steps += 1

        if steps % 20 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'acc_train {:.4f}'.format(acc),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'time: {:.4f}s'.format(time.time() - t))
            wandb.log({'epoch': epoch,
                       'acc_train': acc,
                       'loss_train': loss})

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


def test_model(model, test_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in test_iter:
            text = batch.tweet[0]
            target = batch.organisation
            target = torch.autograd.Variable(target).long()

            prediction = model(text)
            loss = F.cross_entropy(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

        return total_epoch_loss/len(test_iter), total_epoch_acc/len(test_iter)


if __name__ == '__main__':
    DATASET_LOCATION = '/Users/gusy/DOCTORADO/Data/Tweets/mixed.csv'

    model, optimizer = initializate_model()

    wandb.watch(model, log="all")

    train_model(model, optimizer, train_iter, 0)
