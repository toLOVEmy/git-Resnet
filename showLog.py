import matplotlib.pyplot as plt
import pandas as pd

def plot_training_log(log_file):
    # 读取日志文件
    log_data = pd.read_csv(log_file)

    # 获取最大训练和验证准确率及其对应的epoch
    folds = log_data['Fold'].unique()
    max_train_acc = log_data.groupby('Fold')['Training Accuracy'].max()
    max_train_acc_epoch = log_data.loc[log_data.groupby('Fold')['Training Accuracy'].idxmax()].reset_index(drop=True)
    max_val_acc = log_data.groupby('Fold')['Validation Accuracy'].max()
    max_val_acc_epoch = log_data.loc[log_data.groupby('Fold')['Validation Accuracy'].idxmax()].reset_index(drop=True)

    # 打印最大准确率和epoch
    for i, fold in enumerate(folds):
        print(f"Fold {fold}:")
        print(f"  Maximum Training Accuracy: {max_train_acc[fold]} at epoch {max_train_acc_epoch.iloc[i]['Epoch']}")
        print(f"  Maximum Validation Accuracy: {max_val_acc[fold]} at epoch {max_val_acc_epoch.iloc[i]['Epoch']}")

    colors = ['b', 'g', 'r', 'c', 'm']  # 5种不同的颜色

    plt.figure(figsize=(14, 7))

    # 绘制训练和验证损失
    plt.subplot(1, 2, 1)
    for fold, color in zip(folds, colors):
        fold_data = log_data[log_data['Fold'] == fold]
        plt.plot(fold_data['Epoch'], fold_data['Training Loss'], color=color, label=f'Fold {fold} Training Loss')
        plt.plot(fold_data['Epoch'], fold_data['Validation Loss'], color=color, linestyle='--', label=f'Fold {fold} Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制训练和验证准确率
    plt.subplot(1, 2, 2)
    for fold, color in zip(folds, colors):
        fold_data = log_data[log_data['Fold'] == fold]
        plt.plot(fold_data['Epoch'], fold_data['Training Accuracy'], color=color, label=f'Fold {fold} Training Accuracy')
        plt.plot(fold_data['Epoch'], fold_data['Validation Accuracy'], color=color, linestyle='--', label=f'Fold {fold} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    log_file = 'training_log_50_16_yuxian_2_fold.txt'
    plot_training_log(log_file)
