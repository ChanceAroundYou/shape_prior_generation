import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


def setup_seeds(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_dataloader(input_tensor, batch_size, train_ratio=0.9, dtype=torch.float32):
    """创建训练和验证数据加载器"""
    dataset_size = len(input_tensor)
    train_size = int(dataset_size * train_ratio)
    
    indices = np.arange(dataset_size)
    train_indices = torch.from_numpy(np.random.choice(indices, train_size, replace=False))
    val_indices = torch.from_numpy(np.setdiff1d(indices, train_indices.numpy()))
    print(f"train index: {train_indices}")
    print(f"val index: {val_indices}")
    
    train_data = TensorDataset(input_tensor[train_indices].to(dtype))
    val_data = TensorDataset(input_tensor[val_indices].to(dtype))
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_vae(
    model,
    input_tensor,
    input_dim,
    batch_size,
    num_epochs,
    learning_rate,
    kl_rate,
    device,
    tb_path,
    tb_comment=None,
    train_ratio=1,
    seed=717,
    dtype=torch.float32,
    model_save_path=None,
    model_load_path=None,
    optimizer=None,
    scheduler=None,
    skip_train=False,
    val_interval=10,
):
    """通用VAE训练函数"""
    # 设置随机种子
    setup_seeds(seed)

    # 准备数据加载器
    train_loader, val_loader = setup_dataloader(input_tensor, batch_size, train_ratio=train_ratio, dtype=dtype)

    # 初始化模型
    model = model.to(device).to(dtype)
    if model_load_path:
        model.load_state_dict(torch.load(model_load_path, weights_only=True))
    else:
        model.initial()

    # 设置优化器
    if optimizer is None:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.5, 0.999)
        )
        
    # TensorBoard
    tb_path = f"{tb_path}{tb_comment or time.strftime('_%Y-%m-%d_%H:%M:%S', time.localtime())}"
    writer = SummaryWriter(tb_path)
    model_save_path = model_save_path or f"{tb_path}/model.pth"
    model.save_path = tb_path
    # 训练循环变量
    loader_size = len(train_loader)
    val_loader_size = len(val_loader)
    loss_list_dict = {}
    best_loss = None

    if not skip_train:
        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            for i, [input_data] in enumerate(train_loader):
                input_data = input_data.to(device, dtype=dtype).view(input_data.shape[0], input_dim)

                # 前向传播
                x_reconstructed, mu, logvar = model(input_data)

                # 计算损失
                loss_dict = model.loss(x_reconstructed, input_data, mu, logvar, kl_rate)

                # 反向传播
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                optimizer.step()

                # 更新学习率
                if scheduler:
                    scheduler.step()

                # 记录训练损失
                for k, v in loss_dict.items():
                    if k not in loss_list_dict:
                        loss_list_dict[k] = np.zeros(loader_size)
                    loss_list_dict[k][i] = v.item()

            # 每10个epoch验证一次
            if epoch % val_interval == 0 and train_ratio < 1:
                model.eval()
                val_loss_list_dict = {}
                
                with torch.no_grad():
                    for j, [input_data] in enumerate(val_loader):
                        input_data = input_data.to(device, dtype=dtype).view(input_data.shape[0], input_dim)
                        x_reconstructed, mu, logvar = model(input_data)
                        val_loss_dict = model.loss(x_reconstructed, input_data, mu, logvar, kl_rate)
                        
                        for k, v in val_loss_dict.items():
                            if k not in val_loss_list_dict:
                                val_loss_list_dict[k] = np.zeros(val_loader_size)
                            val_loss_list_dict[k][j] = v.item()

                # 记录训练和验证损失到TensorBoard
                for k in loss_list_dict.keys():
                    writer.add_scalars(
                        k,
                        {
                            'train': loss_list_dict[k].mean(),
                            'val': val_loss_list_dict[k].mean()
                        },
                        epoch
                    )

                print(
                    f"Val Epoch {epoch}/{num_epochs} | "
                    + ", ".join([f"{k}: {v.mean():.4f}" for k, v in val_loss_list_dict.items()])
                )
            else:
                for k in loss_list_dict.keys():
                    writer.add_scalars(
                        k,
                        {
                            'train': loss_list_dict[k].mean()
                        },
                        epoch
                    )

            # 保存最佳模型
            if best_loss is None or best_loss["loss"].item() > loss_dict["loss"].item():
                best_loss = loss_dict
                print(
                    f"Best epoch {epoch}/{num_epochs} | "
                    + ", ".join([f"{k}: {v.mean():.4f}" for k, v in loss_list_dict.items()])
                )
                best_params = deepcopy(model.state_dict())

            # 定期打印
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    + ", ".join([f"{k}: {v.mean():.4f}" for k, v in loss_list_dict.items()])
                )

        # 保存最佳模型
        model.load_state_dict(best_params)
        torch.save(best_params, model_save_path)
        writer.close()

    return model, best_loss