import torch.nn as nn
from torch.autograd import Variable
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class, batch_size,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.batch_size = batch_size

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(
            output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=False)
        self.rnn2 = nn.LSTM(rnn_hidden, rnn_hidden, bidirectional=False)

        self.dense = nn.Linear(rnn_hidden, num_class)

        self.rnn_hidden_size = rnn_hidden

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel,
                          kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(
                0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        # h_0 = torch.zeros(1,seq.size(1),self.rnn_hidden_size).requires_grad_().to(device)
        # c_0 = torch.zeros(1,seq.size(1),self.rnn_hidden_size).requires_grad_().to(device)

        # recurrent, (h_out, _) = self.rnn1(seq,(h_0.detach(), c_0.detach()))
        # recurrent, _ = self.rnn2(recurrent)

        # recurrent, (h_out, _) = self.rnn1(seq)
        # h_out = h_out.view(-1, self.rnn_hidden_size)

        lstm_out, self.hidden = self.rnn1(seq, self.hidden)
        h_out = lstm_out[-1].view(self.batch_size, -1)
        output = self.dense(h_out)
        return output  # shape: (seq_len, batch, num_class)

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        return (Variable(torch.zeros(1, batch_size, self.rnn_hidden_size).to(device)),
                Variable(torch.zeros(1, batch_size, self.rnn_hidden_size).to(device)))
