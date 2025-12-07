import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
from tqdm import tqdm
import json
import os


class VideoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, label = self.data[idx]
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2) / 255.0
        label = torch.LongTensor([label]).squeeze()
        return frames, label


class CNNRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(CNNRNN, self).__init__()

        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # RNN部分
        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, channels, time_steps, h, w = x.size()

        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(-1, channels, h, w)

        features = self.cnn(x)
        features = features.view(batch_size, time_steps, -1)

        rnn_out, _ = self.rnn(features)
        last_output = rnn_out[:, -1, :]

        return self.classifier(last_output)


def save_training_history(history, filename='training_history.json'):
    """保存训练历史"""
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存: {filename}")


def load_training_history(filename='training_history.json'):
    """加载训练历史"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001,
                checkpoint_path='best_cnn_rnn_model.pth', continue_training=True):
    """
    训练模型，可继续从之前的检查点训练
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
                # 只保留历史记录，不重新训练
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
        for batch_idx, (inputs, labels) in enumerate(pbar):
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

        # 学习率调整
        scheduler.step(val_acc)

        print(f'\rEpoch {epoch + 1:2d}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_acc:6.2f}%, Val Acc: {val_acc:6.2f}%', end='')

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
            print(f' ✓ (保存最佳模型)')
        else:
            print()

    # 保存训练历史
    save_training_history(history)

    print(f"\n训练完成，最终最佳验证准确率: {best_acc:.2f}%")

    # 返回最终的模型和最佳准确率
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
    train_dataset = VideoDataset(train_data)
    test_dataset = VideoDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # 创建模型
    model = CNNRNN(num_classes=len(class_names))
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # 训练模型（继续训练）
    continue_training = True
    trained_model, best_acc = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=50,  # 总epoch数，从之前的进度开始计数
        continue_training=continue_training
    )

    print(f"最终模型准确率: {best_acc:.2f}%")