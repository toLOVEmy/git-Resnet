import torch
import torch.nn as nn
from transformers import ViTModel


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


# 示例使用
model = CustomViT()
dummy_input = torch.randn(1, 50, 768)  # Batch size = 1, sequence length = 50, embedding dimension = 768
outputs = model(dummy_input)
print(outputs.shape)  # 应输出 torch.Size([1, 768])
