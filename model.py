import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import ViTModel, ViTConfig


# 定义 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# 定义 Bottleneck Block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# 定义 ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Removed avgpool and fc layers

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


class CustomViT(nn.Module):
    def __init__(self):
        super(CustomViT, self).__init__()

        # 使用原始的 ViT Encoder 部分
        self.encoder = ViTModel.from_pretrained('vit_model').encoder

        # 可以选择性地保留 Pooler 层，如果你打算用它来提取特定的特征
        self.pooler = ViTModel.from_pretrained('vit_model').pooler

    def forward(self, x):
        # 直接使用 encoder 部分处理嵌入
        encoder_outputs = self.encoder(x)

        # 取 class token 输出，通常为 encoder_outputs.last_hidden_state[:, 0]
        last_hidden_state = encoder_outputs.last_hidden_state
        class_token_output = last_hidden_state[:, 0]

        # 如果需要使用 pooler 层，确保输入维度正确
        # pooled_output = self.pooler(class_token_output)  # 仅当需要使用 pooler 时

        return class_token_output


class ResNet50WithTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet50WithTransformer, self).__init__()
        self.resnet50 = resnet50()  # Set pretrained=False, we'll load weights manually

        # 添加一个适配层，将特征维度从 2048 映射到 768
        self.feature_extractor = nn.Linear(1024, 768)

        # 类别 token 和位置嵌入
        self.class_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 197, 768))  # 197 = 1 (class_token) + 196 (feature positions)

        # 使用本地的 ViT 模型
        self.vit = CustomViT()

        # 最终的全连接层
        self.fc = nn.Linear(768, num_classes)

        # 定义 Dropout 层
        self.dropout = nn.Dropout(p=0.1)

        # self._init_weights()

        # Load the pretrained weights for ResNet50
        # self._load_pretrained_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def _load_pretrained_weights(self):
        # Load the pretrained weights
        pretrained_weights_path = 'resnet50-19c8e357.pth'
        pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')

        # Remove the fully connected layer weights from the pretrained state dict
        # state_dict = {k: v for k, v in pretrained_weights.items() if not k.startswith('fc.')}
        state_dict = {k: v for k, v in pretrained_weights.items() if not k.startswith(('layer4.', 'fc.'))}

        # Load the weights into the model
        self.resnet50.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.resnet50(x)  # 输出: torch.Size([B, 2048, 7, 7])

        B, C, H, W = x.shape

        # 展平特征图到一维
        x = x.view(B, C, -1).permute(0, 2, 1)  # Reshape to (B, 49, 2048)

        # 将特征维度从 2048 映射到 768
        x = self.feature_extractor(x)  # 变换到 (B, 49, 768)

        # Add class token
        class_token = self.class_token.expand(B, -1, -1)

        x = torch.cat((class_token, x), dim=1)

        x = x + self.position_embedding[:, :x.size(1), :].detach()  # detach position_embedding

        # Apply Dropout only in training mode
        # x = self.dropout(x)

        # 使用本地模型
        # 注意，这里传入的是 `pixel_values`，需要将输入转为符合 ViT 模型要求的格式
        outputs = self.vit(x)

        x = outputs

        x = self.fc(x)
        return x


# Test the forward pass with a dummy input
if __name__ == '__main__':
    model = ResNet50WithTransformer(num_classes=5).to('cuda:0' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
    output = model(dummy_input)
    print(output.shape)  # Expected output shape: torch.Size([1, 5])