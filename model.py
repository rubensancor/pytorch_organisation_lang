import torch
import torch.nn as nn
import torch.nn.functional as F

cpu = torch.device('cpu')
cuda = torch.device('cuda')


class MultiCNN(nn.Module):
    def __init__(self, dropout_prob, embedding_matrix, vocab_size,
                 embedding_length, kernel_heights, in_channels,
                 out_channels, num_labels, dense1_size, dense2_size, users,
                 mixed_memory=False, freeze_embeddings=False):
        super(MultiCNN, self).__init__()

        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.kernel_heights = kernel_heights
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_labels = num_labels
        self.mixed_memory = mixed_memory
        self.freeze = freeze_embeddings
        self.out_channels = out_channels
        self.dense1_size = dense1_size
        self.dense2_size = dense2_size
        self.users = users

        if mixed_memory:
            self.__mixed_model__()
        else:
            self.__default_model__()

    def __default_model__(self):
        # Embedding layer
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
        self.embeddings.from_pretrained(self.embedding_matrix)
        if self.freeze:
            self.embeddings.weight.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                out_channels=self.out_channels,
                                kernel_size=(kernel_h, self.embedding_length))
                                for kernel_h in self.kernel_heights])
        (nn.init.xavier_uniform_(conv.weight) for conv in self.convs)
        

        self.dense1 = nn.Linear(len(self.kernel_heights) * self.out_channels,
                                self.dense1_size)
        nn.init.xavier_uniform_(self.dense1.weight)
        self.dropout1 = nn.Dropout(self.dropout_prob)

        self.dense2 = nn.Linear(self.dense1_size, self.dense2_size)
        nn.init.xavier_uniform_(self.dense2.weight)
        self.dropout2 = nn.Dropout(self.dropout_prob)

        self.dense_soft = nn.Linear(self.dense2_size, self.num_labels)
        nn.init.xavier_uniform_(self.dense_soft.weight)


    def __mixed_model__(self):
        # Embedding layer
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_length).to(cpu)
        self.embeddings.from_pretrained(self.embedding_matrix).to(cpu)
        if self.freeze:
            self.embeddings.weight.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                out_channels=self.out_channels,
                                kernel_size=(kernel_h, self.embedding_length)).to(cuda)
                                for kernel_h in self.kernel_heights])
        (nn.init.xavier_uniform_(conv.weight) for conv in self.convs)

        self.dense1 = nn.Linear(len(self.kernel_heights) * self.out_channels,
                                self.dense1_size).to(cuda)
        nn.init.xavier_uniform_(self.dense1.weight)
        self.dropout1 = nn.Dropout(self.dropout_prob).to(cuda)

        self.dense2 = nn.Linear(self.dense1_size, self.dense2_size).to(cuda)
        self.dropout2 = nn.Dropout(self.dropout_prob).to(cuda)
        nn.init.xavier_uniform_(self.dense2.weight)

        self.dense_soft = nn.Linear(self.dense2_size, self.num_labels).to(cuda)
        nn.init.xavier_uniform_(self.dense_soft.weight)

    def forward(self, x):
        x = self.embeddings(x)

        if self.mixed_memory:
            x = x.to(cuda)

        x = x.unsqueeze(1)

        maxed = [self.conv_block(x, conv) for conv in self.convs]

        x = torch.cat(maxed, dim=1)

        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.dense_soft(x)
        
        if self.users:
            return x
        else:
            return F.log_softmax(x, dim=1)

    def conv_block(self, input, conv_layer):

        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)

        return max_out
