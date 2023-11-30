import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class BasicBlock(nn.Module):
    def __init__(self, in1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in1, in1 * 2, kernel_size=1,
                               stride=1, padding=0, bias=False)
        
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in1*2, in1, kernel_size=3,
                        stride=1, padding=1, bias=False)
        
        self.relu2 = nn.LeakyReLU(0.2)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        
        out = self.relu1(out)

        out = self.conv2(out)
        
        out = self.relu2(out)

        out = residual + 0.1 * out
        return out

        return self.main(input)



class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        #input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        #input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  
        out = out.view(*input.shape)

        return self.gamma * out + input



class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.embed = nn.Embedding(num_classes, num_features * 2)
    self.embed.weight.data[:, :num_features].normal_(1, 0.02)  
    self.embed.weight.data[:, num_features:].zero_()  

  def forward(self, x, y):
    out = self.bn(x)
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out


