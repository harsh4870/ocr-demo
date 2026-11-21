import torch
from torch import nn

IMG_HEIGHT = 32


def conv_block(c_in, c_out, pool):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class CRNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(1, 32, (2, 2)),
            conv_block(32, 64, (2, 2)),
            conv_block(64, 128, (2, 1)),
            conv_block(128, 256, (2, 1)),
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x).mean(dim=2)
        f, _ = self.rnn(f.transpose(1, 2))
        return self.fc(f)
