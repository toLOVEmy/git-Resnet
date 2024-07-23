import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from model import resnet50 as creatmodel

# 绘制特征图
def draw_features(width, height, channels, x, savename):
    fig = plt.figure(figsize=(32, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(channels):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
    fig.savefig(savename, dpi=300)
    plt.close()

# 读取模型
def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# 主函数
def predict(model, img_path, savepath):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img).unsqueeze(0)

    if torch.cuda.is_available():
        model.cuda()
        img = img.cuda()

    with torch.no_grad():
        x = model.conv1(img)  # 选择要可视化的层
        x = x.cpu().numpy()

    draw_features(8, 8, x.shape[1], x, f"{savepath}/conv1_features.png")

if __name__ == "__main__":
    model = creatmodel()
    model.fc = torch.nn.Identity()

    trained_model_path = './weights/FBPresNet50_lr8.891777644141941e-05_batch16_epochs20_best.pth'
    model = load_checkpoint(trained_model_path, model)

    img_path = '../../../SCUT-FBP5500_1/SCUT-FBP5500/test/1/79.jpg'
    savepath = './'
    predict(model, img_path, savepath)
