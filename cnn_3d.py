import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import json
import os
from tqdm import tqdm


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 16, 224, 224)):
        super(Simple3DCNN, self).__init__()

        self.input_shape = input_shape  # (channels, depth, height, width)

        # 减小通道数的3D CNN架构
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))  # (batch, 8, 16, 112, 112)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))  # (batch, 16, 8, 56, 56)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))  # (batch, 32, 4, 28, 28)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))  # (batch, 64, 2, 14, 14)
        )

        # 添加全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 计算全连接层输入大小
        self.fc_input_size = self._get_fc_input_size()
        print(f"全连接层输入大小: {self.fc_input_size}")

        # 修改全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_size, 128),  # 使用计算出的特征数量
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _get_fc_input_size(self):
        """计算全连接层输入大小"""
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)  # 创建假数据
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.global_pool(x)
            return int(torch.prod(torch.tensor(x.shape[1:])))  # 计算展平后的特征数

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def save_training_history(history, filename='3dcnn_training_history.json'):
    """保存训练历史"""
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存: {filename}")


def load_training_history(filename='3dcnn_training_history.json'):
    """加载训练历史"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def train_3dcnn(model, train_loader, val_loader, num_epochs=20, lr=0.001,
                checkpoint_path='best_3dcnn_model.pth', continue_training=True):
    """
    训练3DCNN模型，可继续从之前的检查点训练
    continue_training: 是否从之前的检查点继续训练
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)

    # 检查是否有已保存的模型
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'best_val_acc': 0.0}

    if continue_training and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc = checkpoint.get('val_acc', 0.0)
            print(f"加载检查点，从 epoch {start_epoch + 1} 开始继续训练")
            print(f"之前最佳准确率: {best_acc:.2f}%")

            # 加载训练历史
            loaded_history = load_training_history()
            if loaded_history:
                history = loaded_history
        except Exception as e:
            print(f"加载检查点失败，从头开始训练: {e}")

    # 加载完整的训练历史用于记录
    if continue_training:
        full_history = load_training_history()
        if full_history:
            history = full_history
            # 确保我们不会覆盖已有的历史记录
            if start_epoch > len(history.get('train_loss', [])):
                start_epoch = len(history.get('train_loss', []))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 加载优化器状态
    if continue_training and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("已加载优化器状态")
        except:
            pass

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        history['val_acc'].append(val_acc)
        scheduler.step(val_acc)

        # 打印epoch结果
        epoch_info = f"Epoch {epoch + 1:2d}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:6.2f}%, Val Acc: {val_acc:6.2f}%"

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            history['best_val_acc'] = best_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"{epoch_info} ✓ (保存最佳模型)")
        else:
            print(epoch_info)

    # 保存历史记录
    save_training_history(history)

    print(f"\n训练完成，最佳验证准确率: {best_acc:.2f}%")
    return model, best_acc


if __name__ == "__main__":
    # 加载数据
    with open('processed_data/processed_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    train_data = data_dict['train_data']
    test_data = data_dict['test_data']
    class_names = data_dict['class_names']

    print("数据集信息:")
    print(f"  训练集大小: {len(train_data)}")
    print(f"  测试集大小: {len(test_data)}")
    print(f"  类别数: {len(class_names)}")
    print(f"  类别名称: {class_names}\n")

    # 创建数据加载器
    from cnn_rnn import VideoDataset

    train_dataset = VideoDataset(train_data)
    test_dataset = VideoDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # 创建模型
    model = Simple3DCNN(num_classes=len(class_names))
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # 训练模型
    continue_training = True
    trained_model, best_acc = train_3dcnn(
        model,
        train_loader,
        test_loader,
        num_epochs=50,
        lr=0.0002,
        continue_training=continue_training
    )
    print(f"最终模型准确率: {best_acc:.2f}%")