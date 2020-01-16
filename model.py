import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCNN(nn.Module):
    def __init__(self, dropout_prob, embedding_matrix, vocab_size,
                 embedding_length, kernel_heights, in_channels,
                 out_channels, num_labels, freeze=False):
        super(MultiCNN, self).__init__()

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_length)
        self.embeddings.from_pretrained(embedding_matrix, freeze=freeze)
        
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               (kernel_heights[0], embedding_length))
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               (kernel_heights[1], embedding_length))
        self.conv3 = nn.Conv2d(in_channels, out_channels,
                               (kernel_heights[2], embedding_length))
        self.conv4 = nn.Conv2d(in_channels, out_channels,
                               (kernel_heights[3], embedding_length))

        self.dense1 = nn.Linear(len(kernel_heights) * out_channels, 1024)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dense2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.dense_soft = nn.Linear(256, num_labels)

    def forward(self, x):
        x = self.embeddings(x)
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
