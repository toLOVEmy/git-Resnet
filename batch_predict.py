import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = r'E:\xwc\pycharm_virtual_cnn\SCUT-FBP5500_1\SCUT-FBP5500\test'
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    # img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    img_path_list = []  # 存储所有图片的路径
    class_list = {}  # 存储每个类别的图片路径

    # 获取imgs_root下的所有子文件夹
    subdirs = [f.name for f in os.scandir(imgs_root) if f.is_dir()]

    # 遍历子文件夹
    for subdir in subdirs:
        # 构建子文件夹的完整路径
        subdir_path = os.path.join(imgs_root, subdir)
        # 获取子文件夹下的所有.jpg图片路径
        images = [os.path.join(subdir_path, i) for i in os.listdir(subdir_path) if i.endswith(".jpg")]
        # 将图片路径添加到img_path_list
        img_path_list.extend(images)
        # 将图片路径存储在class_list中，以子文件夹名为键
        class_list[subdir] = images

    # 现在img_path_list包含了所有图片的路径，而class_list是一个字典，键为类别名，值为该类别的图片路径列表

    # read class_indict
    json_path = r'E:\xwc\pycharm_virtual_cnn\deep-learning-for-image-processing-master' \
                r'\pytorch_classification\Test5_resnet\class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet50(num_classes=5).to(device)

    # load model weights
    weights_path = r'E:\xwc\pycharm_virtual_cnn\SCUT-FBP5500_1\SCUT-FBP5500\FBPresNet34.pth'
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    correct_count = 0  # 用于累计预测正确的数量
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            # 使用torch.stack()函数将img_list中的所有图像堆叠成一个批次，dim=0表示在第一个维度（批次维度）上堆叠
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                img_path = img_path_list[ids * batch_size + idx]
                # 根据图片路径找到其在class_list中的实际类别
                actual_class = None
                for class_name, images in class_list.items():
                    if img_path in images:
                        actual_class = class_name
                        break
                # 比较预测类别和实际类别
                if actual_class and class_indict[str(cla.numpy())] == actual_class:
                    correct_count += 1
                    correct_str = "正确"
                else:
                    correct_str = "错误"
                """
                # 打印信息
                print("image: {}  预测类别: {}  实际类别: {}  最大概率: {:.3}  预测是否正确: {}".format(
                    img_path, class_indict[str(cla.numpy())], actual_class, pro.numpy(), correct_str
                ))
                """
        # 打印总的预测正确数量
        print("\n总测试数量有：{}，总的预测正确数量: {},准确率为：{}".format(len(img_path_list), correct_count, correct_count/len(img_path_list)))




if __name__ == '__main__':
    main()
