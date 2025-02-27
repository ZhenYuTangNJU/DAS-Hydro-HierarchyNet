# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:22:23 2024

@author: admin
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib
import os
import warnings

warnings.filterwarnings('ignore')  # 可选

matplotlib.rcParams['font.family'] = 'Times New Roman'

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Hyper parameters
num_epochs_binary = 300   # 二分类的训练轮数
num_epochs_4class = 300   # 四分类的训练轮数
batch_size = 100
learning_rate = 0.001

# 数据路径请根据需要自行修改
data_path = r"C:\Users\admin\Desktop\广州五分类训练数据"

# 读取数据
df_0 = pd.read_excel(os.path.join(data_path, "无水.xlsx"), header=None)
psd_db_array0 = df_0.to_numpy()

df_1 = pd.read_excel(os.path.join(data_path, "1.xlsx"), header=None)
psd_db_array1 = df_1.to_numpy()

df_2 = pd.read_excel(os.path.join(data_path, "2.xlsx"), header=None)
psd_db_array2 = df_2.to_numpy()

df_3 = pd.read_excel(os.path.join(data_path, "3.xlsx"), header=None)
psd_db_array3 = df_3.to_numpy()

df_4 = pd.read_excel(os.path.join(data_path, "4.xlsx"), header=None)
psd_db_array4 = df_4.to_numpy()

def split_data(psd_data, test_size=0.3, val_size=0.5, random_state=48):
    X_train, X_temp = train_test_split(psd_data, test_size=test_size, random_state=random_state)
    X_val, X_test = train_test_split(X_temp, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test

X_train0, X_val0, X_test0 = split_data(psd_db_array0)
X_train1, X_val1, X_test1 = split_data(psd_db_array1)
X_train2, X_val2, X_test2 = split_data(psd_db_array2)
X_train3, X_val3, X_test3 = split_data(psd_db_array3)
X_train4, X_val4, X_test4 = split_data(psd_db_array4)

print("X_train0 set size:", X_train0.shape)
print("X_val0 set size:", X_val0.shape)
print("X_test0 set size:", X_test0.shape)
print("X_train1 set size:", X_train1.shape)
print("X_val1 set size:", X_val1.shape)
print("X_test1 set size:", X_test1.shape)
print("X_train2 set size:", X_train2.shape)
print("X_val2 set size:", X_val2.shape)
print("X_test2 set size:", X_test2.shape)
print("X_train3 set size:", X_train3.shape)
print("X_val3 set size:", X_val3.shape)
print("X_test3 set size:", X_test3.shape)
print("X_train4 set size:", X_train4.shape)
print("X_val4 set size:", X_val4.shape)
print("X_test4 set size:", X_test4.shape)


class CustomDataset(Dataset):
    def __init__(self, data_list, label_list):
        # data_list: [array0, array1, ...]
        # label_list: 对应每组数据的标签
        self.data = []
        self.labels = []
        for data_array, label in zip(data_list, label_list):
            temp_data = torch.tensor(data_array, dtype=torch.float32)
            temp_labels = torch.full((len(temp_data),), label, dtype=torch.long)
            self.data.append(temp_data)
            self.labels.append(temp_labels)

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.unsqueeze(self.data[idx], 0)  
        return sample, self.labels[idx]


# =========== 简单的一维CNN用于二分类模型 ===========

class Simple1DCNNBinary(nn.Module):
    def __init__(self, num_classes=2):
        super(Simple1DCNNBinary, self).__init__()
        # 假设输入为(N, 1, 154)
        # 一个模块：两次卷积+CBAM+残差
        # 我们需要先将residual变成同样通道数(1->64)
        self.res_adjust = nn.Conv1d(1, 64, kernel_size=1, stride=1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
        )
        self.cbam = cbamblock(channel=64, ratio=16, kernel_size=7)

        # MaxPool后 (N,64,77)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) # 154->77
        self.fc = nn.Linear(64*77, num_classes)

    def forward(self, x):
        # 残差分支
        residual = x
        residual = self.res_adjust(residual)  # (N,64,154)

        x = self.conv1(x)  # (N,64,154)
        x = self.conv2(x)  # (N,64,154)
        x = self.cbam(x)   # (N,64,154)

        x = x + residual
        x = torch.tanh(x)

        x = self.pool(x)   # (N,64,77)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# =========== 复杂模型用于四分类（原有的Inception+CBAM模型） ===========
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv1d(in_channels, c1, kernel_size=1)

        self.p2_1 = nn.Conv1d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv1d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv1d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv1d(c3[0], c3[1], kernel_size=5, padding=2)
        
        self.p4_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv1d(in_channels, c4, kernel_size=1)
        
        self.residual_conv = nn.Conv1d(in_channels, c1+c2[1]+c3[1]+c4, kernel_size=1)   
        
        self.Tanh = nn.Tanh()

    def forward(self, x):
        p1 = torch.tanh(self.p1_1(x))
        p2 = torch.tanh(self.p2_2(torch.tanh(self.p2_1(x))))
        p3 = torch.tanh(self.p3_2(torch.tanh(self.p3_1(x))))
        p4 = torch.tanh(self.p4_2(self.p4_1(x)))
        
        inception_out = torch.cat((p1, p2, p3, p4), dim=1)
        residual_out = self.residual_conv(x)
        
        return self.Tanh(inception_out + residual_out)


class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  
        self.max_pool = nn.AdaptiveMaxPool1d(1)  
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = avg_out
        return self.sigmoid(out)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x = torch.cat([avg_out, max_out], dim=1)  
        x = self.conv1(x)  
        return self.sigmoid(x)


class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbamblock, self).__init__()
        self.channelattention = ChannelAttention1D(channel, ratio=ratio)
        self.spatialattention = SpatialAttention1D(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  
        x = x * self.spatialattention(x)  
        return x


class ConvNetWithSelfAttentionAndResidual(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNetWithSelfAttentionAndResidual, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2) # 154 -> 77
        )
        
        self.Inception1 = Inception(128, 64, (96,128), (16,32), 32)     
        self.cbam_Inception1 = cbamblock(channel=256, ratio=16, kernel_size=7) 
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Tanh()
        )
        self.layer3_adjust = nn.Conv1d(256, 128, kernel_size=1, stride=1)  
        self.cbam1 = cbamblock(channel=128, ratio=16, kernel_size=7)  

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Tanh()
        )
        
        self.layer5_adjust = nn.Conv1d(128, 128, kernel_size=1, stride=1) 
        self.cbam2 = cbamblock(channel=128, ratio=16, kernel_size=7)
        
        self.layer5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.Tanh()
        )
        
        self.layer7_adjust = nn.Conv1d(128, 256, kernel_size=1, stride=1) 
        self.cbam3 = cbamblock(channel=256, ratio=16, kernel_size=7)  
        
        self.layer7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
        )
        self.Inception2 = Inception(256, 64, (96,128), (16,32), 32)  
        self.cbam_Inception2 = cbamblock(channel=256, ratio=16, kernel_size=7) 

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) # 77 -> 38
        self.fc = nn.Linear(256*38, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.Inception1(out)
        out = self.cbam_Inception1(out)
        residual1 = out
        residual1 = self.layer3_adjust(residual1)
        
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.cbam1(out)
        out = out + residual1  
        out = torch.tanh(out)
        residual2 = out 
        
        out = self.layer4(out)
        out = self.layer5(out)
        residual2 = self.layer5_adjust(residual2)
        out = self.cbam2(out)
        out = out + residual2 
        out = torch.tanh(out)
        residual3 = out 
        
        out = self.layer6(out)
        out = self.layer7(out)
        residual3 = self.layer7_adjust(residual3)
        out = self.cbam3(out)
        out = out + residual3
        out = torch.tanh(out)
        
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, num_classes, model_save_path, prefix='model'):
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    early_stopping_patience = 20
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    best_epoch = 0       
    best_accuracy = 0.0       
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        epoch_train_losses = []
        model.train()
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for j, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        # 早停和保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1       
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            early_stopping_counter = 0
            print(f'Model improved and saved at epoch {epoch + 1}, val loss {best_val_loss:.4f}, acc {best_accuracy:.4f}')
        else:
            early_stopping_counter += 1
            print(f'No improvement. Early stopping counter {early_stopping_counter}/{early_stopping_patience}')
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f'Best model saved at epoch {best_epoch}, val loss {best_val_loss:.4f}, val acc {best_accuracy:.4f}')

    # 测试集评估
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_preds = []
    all_labels = []
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(probabilities) 
            all_labels.extend(labels.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    correct = (np.array(y_pred) == np.array(y_true)).sum()
    total = len(y_true)
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # 绘图和保存指标
    sns.set(style='whitegrid', palette='muted', color_codes=True)
    
    # Loss曲线
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='deepskyblue', linewidth=2.5, linestyle='-', marker='o', markersize=8, alpha=0.9)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='salmon', linewidth=2.5, linestyle='--', marker='s', markersize=8, alpha=0.9)
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.legend(prop={'size': 20})
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=20)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')
    plt.savefig(f'{prefix}_loss_curve.png', bbox_inches='tight')
    plt.show()

    # Accuracy曲线
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='skyblue', linewidth=2.5, linestyle='-', marker='o', markersize=8, alpha=0.9)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='mediumseagreen', linewidth=2.5, linestyle='-', marker='^', markersize=8, alpha=0.9)
    plt.xlabel('Epoch', fontweight='bold', fontsize=22)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=22)
    plt.legend(prop={'size': 20, 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=20)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontweight('bold')
    plt.savefig(f'{prefix}_accuracy_curve.png', bbox_inches='tight')
    plt.show()

    # 混淆矩阵
    plt.figure(figsize=(12, 8), dpi=300)
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 22}, cbar=False, linecolor='black', linewidths=1)
    plt.xlabel('Predicted Label', fontsize=22)
    plt.ylabel('True Label', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')
    plt.savefig(f'{prefix}_confusion_matrix.png', bbox_inches='tight')
    plt.show()

    # 精确率、召回率、F1
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    plt.figure(figsize=(11.5, 8), dpi=300)
    metrics = [precision, recall, fscore]
    labels = ['Precision', 'Recall', 'F1-Score']
    colors_bar = ['skyblue', 'salmon', 'mediumseagreen']
    bars = plt.bar(labels, metrics, color=colors_bar, edgecolor='black', linewidth=2)
    plt.ylim(0, 1.05)
    plt.tick_params(axis='both', which='major', labelsize=20)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=18, fontweight='bold')
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontweight('bold')
    plt.gca().set_yticklabels(['{:.2f}'.format(x) for x in plt.gca().get_yticks()], fontsize=22)
    plt.savefig(f'{prefix}_metrics_bar.png', bbox_inches='tight')
    plt.show()

    # # ROC曲线和AUC（多分类宏平均）
    # all_preds = np.array(all_preds)
    # y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    # fpr = dict()
    # tpr = dict()
    # roc_auc_dict = dict()

    # for i in range(num_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_preds[:, i])
    #     roc_auc_dict[i] = auc(fpr[i], tpr[i])

    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(num_classes):
    #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # mean_tpr /= num_classes

    # fpr_macro = all_fpr
    # tpr_macro = mean_tpr
    # roc_auc_macro = auc(fpr_macro, tpr_macro)

    # plt.figure(figsize=(8, 6), dpi=300)
    # plt.plot(fpr_macro, tpr_macro, color='darkorange', lw=2, label='Macro-average ROC (AUC = %0.2f)' % roc_auc_macro)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate', fontsize=16)
    # plt.ylabel('True Positive Rate', fontsize=16)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.legend(loc="lower right", fontsize=14)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.savefig(f'{prefix}_roc_curve.png', bbox_inches='tight')
    # plt.show()

    # 保存数据到Excel
    df_losses = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Training Loss': train_losses,
        'Validation Loss': val_losses
    })
    df_losses.to_excel(f'{prefix}_losses.xlsx', index=False)

    df_accuracies = pd.DataFrame({
        'Epoch': range(1, len(train_accuracies) + 1),
        'Training Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies
    })
    df_accuracies.to_excel(f'{prefix}_accuracies.xlsx', index=False)

    df_conf_matrix = pd.DataFrame(conf_matrix)
    df_conf_matrix.to_excel(f'{prefix}_confusion_matrix.xlsx', index=False)

    df_performance = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Value': [precision, recall, fscore]
    })
    df_performance.to_excel(f'{prefix}_performance_metrics.xlsx', index=False)

    # df_roc = pd.DataFrame({
    #     'False Positive Rate': fpr_macro,
    #     'True Positive Rate': tpr_macro
    # })
    # df_roc.to_excel(f'{prefix}_roc_curve.xlsx', index=False)

    print("All processing done!")
    return model


# ====================== 阶段1：二分类 ======================
# 对于二分类，将0类标为0，将(1,2,3,4)合并为1类
X_train_binary = [X_train0, np.concatenate([X_train1, X_train2, X_train3, X_train4], axis=0)]
y_train_binary = [0, 1]
train_dataset_binary = CustomDataset(X_train_binary, y_train_binary)

X_val_binary = [X_val0, np.concatenate([X_val1, X_val2, X_val3, X_val4], axis=0)]
y_val_binary = [0, 1]
val_dataset_binary = CustomDataset(X_val_binary, y_val_binary)

X_test_binary = [X_test0, np.concatenate([X_test1, X_test2, X_test3, X_test4], axis=0)]
y_test_binary = [0, 1]
test_dataset_binary = CustomDataset(X_test_binary, y_test_binary)

train_loader_binary = DataLoader(train_dataset_binary, batch_size=batch_size, shuffle=True)
val_loader_binary = DataLoader(val_dataset_binary, batch_size=batch_size, shuffle=True)
test_loader_binary = DataLoader(test_dataset_binary, batch_size=batch_size, shuffle=False)

model_binary = Simple1DCNNBinary().to(device)
model_binary = train_and_evaluate(model_binary, train_loader_binary, val_loader_binary, test_loader_binary, 
                                  num_epochs=num_epochs_binary, num_classes=2, 
                                  model_save_path='model_best_val_binary.ckpt', prefix='binary_class')


# ====================== 阶段2：四分类 ======================
# 对于四分类使用原复杂模型和之前的一样
X_train_4class = [X_train1, X_train2, X_train3, X_train4]
y_train_4class = [0, 1, 2, 3] # 将原1类对应0，2类对应1，3类对应2，4类对应3
train_dataset_4class = CustomDataset(X_train_4class, y_train_4class)

X_val_4class = [X_val1, X_val2, X_val3, X_val4]
y_val_4class = [0, 1, 2, 3]
val_dataset_4class = CustomDataset(X_val_4class, y_val_4class)

X_test_4class = [X_test1, X_test2, X_test3, X_test4]
y_test_4class = [0, 1, 2, 3]
test_dataset_4class = CustomDataset(X_test_4class, y_test_4class)

train_loader_4class = DataLoader(train_dataset_4class, batch_size=batch_size, shuffle=True)
val_loader_4class = DataLoader(val_dataset_4class, batch_size=batch_size, shuffle=True)
test_loader_4class = DataLoader(test_dataset_4class, batch_size=batch_size, shuffle=False)

model_4class = ConvNetWithSelfAttentionAndResidual(num_classes=4).to(device)
model_4class = train_and_evaluate(model_4class, train_loader_4class, val_loader_4class, test_loader_4class, 
                                  num_epochs=num_epochs_4class, num_classes=4, 
                                  model_save_path='model_best_val_4class.ckpt', prefix='four_class')

print("二分类和四分类模型训练与评估完成！")
