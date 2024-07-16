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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

def train_and_evaluate(params):
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

    batch_size = int(params['batch_size'])
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
    model_weight_path = r'E:\xwc\pycharm_virtual_cnn\deep-learning-for-image-processing-master' \
                        r'\pytorch_classification\Test5_resnet\resnet50-19c8e357.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    params_lr = params['lr']
    optimizer = optim.Adam(net.parameters(), lr=params_lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'], eta_min=0)

    epochs = int(params['epochs'])
    best_acc = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        correct_train = 0

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predict_y = torch.max(logits, dim=1)[1]
            correct_train += torch.eq(predict_y, labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_acc = correct_train / train_num

        net.eval()
        acc = 0.0
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
        running_loss = running_loss / len(train_loader)
        running_val_loss = running_val_loss / len(validate_loader)

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss, running_val_loss, train_acc, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f'FBPresNet50_lr{params_lr}_batch{batch_size}_epochs{epochs}_best.pth'
            torch.save(net.state_dict(), save_path)

        scheduler.step()

    print('Finished Training')

    return {'loss': -best_acc, 'status': STATUS_OK}

def hyperopt_train_test(params):
    result = train_and_evaluate(params)
    return result['loss']

space = {
    'lr': hp.uniform('lr', 1e-5, 1e-3),
    'batch_size': hp.choice('batch_size', [8, 16, 32]),
    'epochs': hp.choice('epochs', [10, 20, 30, 40, 50, 60])
}

trials = Trials()
best = fmin(fn=hyperopt_train_test,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

print("Best parameters found: ", best)
