import torch
from torchtext import data
from torchtext.vocab import Vectors



# TODO: check if the file exists or not to give a exception

def load_dataset(path, device, batch_size, seed):

    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

    tokenize = lambda x: x.split()

    # TODO: change fix_length to the largest string
    TEXT = data.Field(sequential=True,
                      tokenize=tokenize,
                      lower=True,
                      include_lengths=True,
                      batch_first=True,
                      fix_length=100)

    LABEL = data.LabelField(sequential=False)

    dataset = data.TabularDataset(path=path,
                                  format='CSV',
                                  fields=[('organisation', LABEL),
                                          ('tweet', TEXT)])

    (train_data,
     valid_data,
     test_data) = dataset.split(split_ratio=[0.7, 0.1, 0.2],
                                stratified=True,
                                strata_field='organisation')

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
