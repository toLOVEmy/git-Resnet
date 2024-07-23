import torch
import torchvision.models as models

def print_state_dict(state_dict, title):
    """
    打印state_dict的键值对
    """
    print(f"{title}:")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
    print("\n")

def analyze_state_dicts(pretrained_dict, model_dict):
    """
    分析哪些部分的权重可以被载入模型，并指出被载入到模型的哪部分中
    """
    matching_keys = {k: (v.shape, model_dict[k].shape) for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    non_matching_keys = {k: (v.shape, model_dict[k].shape) for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape != v.shape}
    not_in_model_keys = {k: v.shape for k, v in pretrained_dict.items() if k not in model_dict}

    print("可以被载入模型的Keys权重:")
    for key, (pretrained_shape, model_shape) in matching_keys.items():
        print(f"预训练键 {key}: 预训练形状 {pretrained_shape} 可以被载入模型键 {key} 形状 {model_shape}")

    print("\n因为形状不匹配不可以被载入模型的Keys权重:")
    for key, (pretrained_shape, model_shape) in non_matching_keys.items():
        print(f"预训练键 {key}: 预训练形状 {pretrained_shape} 不匹配模型键 {key} 形状 {model_shape}")

    print("\n因为不在模型中不可以被载入模型的Keys权重:")
    for key, pretrained_shape in not_in_model_keys.items():
        print(f"预训练键 {key}: 预训练形状 {pretrained_shape}")

def main(pretrained_weight_path, model_class):
    # 加载预训练权重
    pretrained_dict = torch.load(pretrained_weight_path, map_location='cpu')
    print_state_dict(pretrained_dict, "预训练权重")

    # 初始化模型并获取模型权重
    model = model_class(pretrained=False)  # 初始化模型但不加载预训练权重
    model_dict = model.state_dict()
    print_state_dict(model_dict, "模型权重")

    # 分析哪些部分的权重可以被载入模型
    analyze_state_dicts(pretrained_dict, model_dict)

    # 加载匹配的权重到模型中
    model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape})
    model.load_state_dict(model_dict, strict=False)

if __name__ == '__main__':
    # pretrained_weight_path = '../vision_transformer/jx_vit_base_patch16_224_in21k-e5005f0a.pth'  # 替换为你的预训练权重文件路径
    pretrained_weight_path = 'resnet50-19c8e357.pth'  # 替换为你的预训练权重文件路径
    model_class = models.resnet18  # 替换为你的模型类

    main(pretrained_weight_path, model_class)
