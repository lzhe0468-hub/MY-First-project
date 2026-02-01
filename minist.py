import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
def check_environment():
    print("-" * 30)
    print("正在检查 PyTorch 环境...")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 检查 GPU 设备
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 检测到 NVIDIA 显卡: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 检测到 Apple Silicon (M1/M2/M3) 加速")
    else:
        print("⚠️ 未检测到 GPU，将使用 CPU 运行 (速度较慢)")
    
    print("-" * 30)
    return device

# 定义一个简单的卷积神经网络 (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 1: 输入1通道(黑白图)，输出32通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积层 2: 输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10个数字分类
        # 激活与池化
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 第一层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv1(x)))
        # 第二层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # 1. 环境检查
    device = check_environment()

    # 2. 数据准备 (自动下载 MNIST)
    print("\n正在准备数据 (首次运行会自动下载)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 的标准均值和方差
    ])
    
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        print("✅ 数据集加载成功！")
    except Exception as e:
        print(f"❌ 数据下载失败，请检查网络: {e}")
        return

    # 3. 初始化模型、损失函数、优化器
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. 试运行训练 (只跑 1 个 Epoch 验证环境)
    print(f"\n开始在 {device} 上进行测试训练 (1个 Epoch)...")
    model.train()
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 搬运数据到 GPU
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
            
        # 为了快速验证，我们只跑前 300 个 Batch 就停下
        if batch_idx > 300:
            print("...")
            break

    end_time = time.time()
    print("-" * 30)
    print(f"✅ 测试完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"🎉 恭喜！你的 PyTorch 环境配置正确，可以进行深度学习了。")

if __name__ == '__main__':
    main()

