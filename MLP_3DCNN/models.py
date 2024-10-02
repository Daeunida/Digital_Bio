import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, 128)   # hidden1, hidden2, hidden3, bn1 등의 이름은 사용자가 정의함
        self.bn1 = nn.BatchNorm1d(128)
        self.hidden2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.hidden3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1) # 입력 차원 64차원에서 1로 변환 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.hidden1(x)))
        x = self.dropout(x) # 레이어의 출력 값에서 일부 노드가 비활성화 됨. 
        x = self.relu(self.bn2(self.hidden2(x)))
        x = self.dropout(x) 
        x = self.relu(self.bn3(self.hidden3(x)))
        x = self.dropout(x)
        features = x # 중간 feature 저장하기 위해 
        x = self.output(x)
        return x, features

def conv3d1(inplanes, planes, stride=1): # inplanes : 입력 텐서의 채널 수(흑백일 경우엔 1), planes : 출력 텐서의 채널 수, kernel_size : 1x1x1로 커널 크기 설정
    return nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # 편향 사용하지

def conv3d3(inplanes, planes, stride=1):
    return nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, momentum, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3d3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum=momentum, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3d3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum=momentum, track_running_stats=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, momentum, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3d1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes, momentum=momentum, track_running_stats=True)
        self.conv2 = conv3d3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes, momentum=momentum, track_running_stats=True)
        self.conv3 = conv3d1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion, momentum=momentum, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, momentum, num_classes=1):
        n_inp = 1
        self.inplanes = n_inp
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes, momentum=momentum, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.layer1 = self._make_layer(block, n_inp, layers[0], momentum)
        self.layer2 = self._make_layer(block, 2 * n_inp, layers[1], momentum, stride=2)
        self.layer3 = self._make_layer(block, 4 * n_inp, layers[2], momentum, stride=2)
        self.layer4 = self._make_layer(block, 8 * n_inp, layers[3], momentum, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(8 * n_inp * block.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, momentum, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3d1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion, momentum=momentum, track_running_stats=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, momentum, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1, -1)
        x = self.fc(features)
        return x, features  # fc 레이어 전에 feature 반환

def ResNet3d():
    model = ResNet(Bottleneck, [3, 4, 23, 3], momentum=0.1)
    return model
