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

from model import resnet50

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
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
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

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet50()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = r'E:\xwc\pycharm_virtual_cnn\deep-learning-for-image-processing-master' \
                        r'\pytorch_classification\Test5_resnet\resnet50-19c8e357.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict方法载入模型预训练权重
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.000001)

    # 设置余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    epochs = 60
    best_acc = 0.0
    save_path = '.\FBPresNet50_16_yuxian_01.pth'
    train_steps = len(train_loader)
    train_start_time = time.perf_counter()  # 记录训练开始时间，用于计算训练用时
    log_file = 'training_log50_16_yuxian_01.txt'

    # 写入CSV头
    with open(log_file, 'w') as f:
        f.write('Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy\n')

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

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
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

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_acc = acc / val_num
        running_loss = running_loss / train_steps
        running_val_loss = running_val_loss / len(validate_loader)

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss, running_val_loss, train_acc, val_acc))

        # 以CSV格式写入日志文件
        with open(log_file, 'a') as f:
            f.write(f'{epoch + 1},{running_loss:.3f},{running_val_loss:.3f},{train_acc:.3f},{val_acc:.3f}\n')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

        # 更新学习率
        scheduler.step()

    print('Finished Training')

    # 记录结束时间
    train_end_time = time.perf_counter()
    # 计算代码块执行所需的时间
    elapsed_time = train_end_time - train_start_time
    minute_time = elapsed_time // 60
    second_time = elapsed_time % 60
    print(f"代码执行耗时：{elapsed_time}秒")
    print(f"代码执行耗时：{minute_time}分{second_time}秒")


if __name__ == '__main__':
    main()
