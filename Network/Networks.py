from torch import nn

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.name = "CNN"
    out1 = 32
    out2 = 64
    out3 = 128
    out4 = 64
    out5 = 16
    out6 = 2
    self.cnn_layers = nn.Sequential(
      # Layer 1
      nn.Conv3d(1,out1,2,2),
      nn.BatchNorm3d(out1),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      # Layer 2
      nn.Conv3d(out1, out2, 2, 2),
      nn.BatchNorm3d(out2),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      # Layer 3
      nn.Conv3d(out2, out3, 2, 2),
      nn.BatchNorm3d(out3),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      # Layer 4
      nn.Conv3d(out3, out4, 1, 1),
      nn.BatchNorm3d(out4),
      nn.LeakyReLU(inplace=True),
      # Layer 5
      nn.Conv3d(out4, out5, 1, 1),
      nn.BatchNorm3d(out5),
      nn.LeakyReLU(inplace=True),
      # Layer 6
      nn.Conv3d(out5, out6, 1, 1),
      nn.BatchNorm3d(out6),
      nn.LeakyReLU(inplace=True),
      nn.AvgPool3d(2)
    )

  def forward(self, x):
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)
    return x