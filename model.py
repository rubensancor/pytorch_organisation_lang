import torch
import torch.nn as nn
import torch.nn.functional as F

cpu = torch.device('cpu')
cuda = torch.device('cuda')


class MultiCNN(nn.Module):
    def __init__(self, dropout_prob, embedding_matrix, vocab_size,
                 embedding_length, kernel_heights, in_channels,
                 out_channels, num_labels, mixed_memory=False,
                 freeze_embeddings=False):
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

        if mixed_memory:
            self.__mixed_model__()
        else:
            self.__default_model__()

    def __default_model__(self):
        # Embedding layer
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
        self.embeddings.from_pretrained(self.embedding_matrix,
                                        freeze=self.freeze)

        # Conv layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[0], self.embedding_length))
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[1], self.embedding_length))
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[2], self.embedding_length))
        self.conv4 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[3], self.embedding_length))

        self.dense1 = nn.Linear(len(self.kernel_heights) * self.out_channels,
                                1024)
        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.dense2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(self.dropout_prob)

        self.dense_soft = nn.Linear(256, self.num_labels)

    def __mixed_model__(self):
        # Embedding layer
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_length).to(cpu)
        self.embeddings.from_pretrained(self.embedding_matrix,
                                        freeze=self.freeze).to(cpu)

        # Conv layers
        self.conv1 = nn.Conv2d(self.in_channels,
                               self.out_channels,
                               (self.kernel_heights[0],
                                self.embedding_length)).to(cuda)
        self.conv2 = nn.Conv2d(self.in_channels,
                               self.out_channels,
                               (self.kernel_heights[1],
                                self.embedding_length)).to(cuda)
        self.conv3 = nn.Conv2d(self.in_channels,
                               self.out_channels,
                               (self.kernel_heights[2],
                                self.embedding_length)).to(cuda)
        self.conv4 = nn.Conv2d(self.in_channels,
                               self.out_channels,
                               (self.kernel_heights[3],
                                self.embedding_length)).to(cuda)

        self.dense1 = nn.Linear(len(self.kernel_heights) * self.out_channels,
                                1024).to(cuda)
        self.dropout1 = nn.Dropout(self.dropout_prob).to(cuda)
        self.dense2 = nn.Linear(1024, 256).to(cuda)
        self.dropout2 = nn.Dropout(self.dropout_prob).to(cuda)

        self.dense_soft = nn.Linear(256, self.num_labels).to(cuda)

    def forward(self, x):
        x = self.embeddings(x)

        if self.mixed_memory:
            x = x.to(cuda)

        x = x.unsqueeze(1)

        max_out1 = self.conv_block(x, self.conv1)
        max_out2 = self.conv_block(x, self.conv2)
        max_out3 = self.conv_block(x, self.conv3)
        max_out4 = self.conv_block(x, self.conv4)

        x = torch.cat((max_out1, max_out2, max_out3, max_out4), 1)

        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)

        x = self.dense_soft(x)

        return x

    def conv_block(self, input, conv_layer):

        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)

        return max_out

