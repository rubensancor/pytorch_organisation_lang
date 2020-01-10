import torch
from torchtext import data
from torchtext.vocab import Vectors
from gensim.models.wrappers import FastText
import gensim


def load_dataset(path):

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

    train_data, test_data = dataset.split(split_ratio=0.8, stratified=True,
                                          strata_field='organisation')

    # Load embeddings from gensim 
    # embedding_model = gensim.models.KeyedVectors.load_word2vec_format('6orgs_embeddings.vec')
    # weights = torch.FloatTensor(embedding_model.vectors)
    TEXT.build_vocab(train_data, vectors=Vectors(name='6orgs_embeddings.vec'))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))

    # Further splitting of training_data to create new
    # training_data & validation_data
    train_data, valid_data = train_data.split(split_ratio=0.8, stratified=True,
                                              strata_field='organisation')
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=2048, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
