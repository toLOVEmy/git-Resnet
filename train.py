import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model import resnet34_StoDepth_lineardecay as creatmodel
from torch.utils.tensorboard import SummaryWriter
import torchprofile  # Import torchprofile for FLOPS calculation
from label_smoothing import LabelSmoothingCrossEntropy  # Import the custom Label Smoothing loss

# 设置随机种子
seed = 42
torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
random.seed(seed)  # 设置 Python 内置的随机种子
np.random.seed(seed)  # 设置 numpy 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 设备的随机种子
# 设置 cuDNN 的确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    # 每个 worker 设置相同的随机种子
    np.random.seed(seed)
    random.seed(seed)


def print_model_params(model):
    """ 打印模型参数数量 """
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {num_params:,}")


def print_model_flops(model, input_size=(1, 3, 224, 224)):
    """ 打印模型FLOPS """
    # 使用 torchprofile 计算 FLOPS
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
    flops = torchprofile.profile_macs(model, dummy_input)
    print(f"模型FLOPS: {flops / 1e9:.3f} GFLOPS")  # 将 FLOPS 转换为 GFLOPS


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # get data root path
    image_path = os.path.join(data_root, "SCUT-FBP5500_1", "SCUT-FBP5500")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    batch_size = 16
    lr = 0.0001
    note = 'resnet34_stopth_label_smoothing'
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])  # windows下线程设为0就行，linux下可以设为8
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, worker_init_fn=worker_init_fn)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, worker_init_fn=worker_init_fn)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 1. 定义模型并加载权重
    net = creatmodel()
    model_weight_path = r'./resnet34-333f7ec4.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # 打印模型参数和FLOPS
    print_model_params(net)
    print_model_flops(net, input_size=(1, 3, 224, 224))

    # 2. 记录模型图到 TensorBoard
    tb_writer = SummaryWriter()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 创建一个与输入数据相同形状的虚拟输入
    tb_writer.add_graph(net, dummy_input)  # 记录模型图

    # define loss function with label smoothing
    loss_function = LabelSmoothingCrossEntropy(eps=0.1, reduction='mean')
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    # 设置余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    epochs = 60
    best_acc = 0.0
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    # 记录训练开始时间
    train_start_time = time.perf_counter()

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        correct_train = 0  # accumulate correct predictions for training
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            predict_y = torch.max(logits, dim=1)[1]
            correct_train += torch.eq(predict_y, labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_acc = correct_train / train_num

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        running_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                running_val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_acc = acc / val_num
        running_loss /= len(train_loader)
        running_val_loss /= len(validate_loader)

        # 记录到TensorBoard
        tb_writer.add_scalar('train_loss', running_loss, epoch)
        tb_writer.add_scalar('train_acc', train_acc, epoch)
        tb_writer.add_scalar('val_loss', running_val_loss, epoch)
        tb_writer.add_scalar('val_acc', val_acc, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss, running_val_loss, train_acc, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            model_filename = "./weights/model-{}-lr{}-bs{}-{}.pth".format(epochs, lr, batch_size, note)
            torch.save(net.state_dict(), model_filename)
            print(f"Epoch {epoch}: 保存新最佳模型 {model_filename}，验证准确率: {val_acc:.4f}")

        # 更新学习率
        scheduler.step()

    print('Finished Training')

    # 记录结束时间
    train_end_time = time.perf_counter()
    # 计算训练用时
    elapsed_time = train_end_time - train_start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    print(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

    tb_writer.close()


if __name__ == '__main__':
    main()
