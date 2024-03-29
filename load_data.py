import torch
from torchtext import data
from torchtext.vocab import Vectors
import random
import pandas as pd
# TODO: check if the file exists or not to give a exception


def load_dataset(path, device, batch_size):

    tokenize = lambda x: x.split()

    TEXT = data.Field(sequential=True,
                      tokenize=tokenize,
                      lower=True,
                      include_lengths=True,
                      batch_first=True)

    LABEL = data.LabelField(sequential=False)

    dataset = data.TabularDataset(path=path,
                                  format='CSV',
                                  fields=[('organisation', LABEL),
                                          ('tweet', TEXT)])
    # random.setstate(st)
    (train_data,
     valid_data,
     test_data) = dataset.split(split_ratio=[0.7, 0.1, 0.2],
                                stratified=True,
                                strata_field='organisation',
                                random_state=random.seed(1234))

    TEXT.build_vocab(train_data, vectors=Vectors(name='./embeddings/f_6orgs_embeddings.vec'))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors

    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)), flush=True)
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size(), flush=True)
    print("Label Length: " + str(len(LABEL.vocab)), flush=True)

    train_iter = data.BucketIterator(train_data,
                                     batch_size=batch_size,
                                     sort_key=lambda x: len(x.text),
                                     repeat=False,
                                     shuffle=True,
                                     device=device)

    test_iter = data.BucketIterator(test_data,
                                    batch_size=batch_size,
                                    sort_key=lambda x: len(x.text),
                                    repeat=False,
                                    shuffle=True,
                                    device=device)
    valid_iter = data.BucketIterator(valid_data,
                                     batch_size=batch_size,
                                     sort_key=lambda x: len(x.text),
                                     repeat=False,
                                     shuffle=True,
                                     device=device)

    vocab_size = len(TEXT.vocab)
    label_size = len(LABEL.vocab)

    return (TEXT, vocab_size, label_size, word_embeddings,
            train_iter, valid_iter, test_iter, LABEL.vocab.itos)


def load_dataset_users(train_path, val_path, test_path, device, batch_size):

    tokenize = lambda x: x.split()

    TEXT = data.Field(sequential=True,
                      tokenize=tokenize,
                      lower=True,
                      include_lengths=True,
                      batch_first=True)

    LABEL = data.LabelField(sequential=False)


    train_data = data.TabularDataset(path=train_path,
                                     format='CSV',
                                     fields=[('user_id', None),
                                             ('organisation', LABEL),
                                             ('tweet', TEXT)],
                                     skip_header=True)

    valid_data = data.TabularDataset(path=val_path,
                                     format='CSV',
                                     fields=[('user_id', None),
                                             ('organisation', LABEL),
                                             ('tweet', TEXT)],
                                     skip_header=True)

    test_data = data.TabularDataset(path=test_path,
                                    format='CSV',
                                    fields=[('user_id', None),
                                            ('organisation', LABEL),
                                            ('tweet', TEXT)],
                                    skip_header=True)

    TEXT.build_vocab(train_data, vectors=Vectors(name='./embeddings/f_6orgs_embeddings.vec'))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors

    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)), flush=True)
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size(), flush=True)
    print("Label Length: " + str(len(LABEL.vocab)), flush=True)

    train_iter = data.BucketIterator(train_data,
                                     batch_size=batch_size,
                                     sort_key=lambda x: len(x.text),
                                     repeat=False,
                                     shuffle=True,
                                     device=device)
    valid_iter = data.BucketIterator(valid_data,
                                     batch_size=batch_size,
                                     sort_key=lambda x: len(x.text),
                                     repeat=False,
                                     shuffle=True,
                                     device=device)
    test_iter = data.BucketIterator(test_data,
                                    batch_size=batch_size,
                                    sort_key=lambda x: len(x.text),
                                    repeat=False,
                                    shuffle=False,
                                    device=device)

    vocab_size = len(TEXT.vocab)
    label_size = len(LABEL.vocab)

    return (TEXT, vocab_size, label_size, word_embeddings,
            train_iter, valid_iter, test_iter, LABEL.vocab.itos)
