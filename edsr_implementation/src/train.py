import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import EDSR # Импортируем модель
from dataset import CustomImageDataset
from config import train_hr_data_path, train_lr_data_path, val_hr_data_path, val_lr_data_path

def main():
    """Функция для обучения модели."""

    # Параметры
    scale_factor = 2
    patch_size_lr = 128
    batch_size = 8

    train_dataset = CustomImageDataset(
        hr_dir=train_hr_data_path,
        lr_dir=train_lr_data_path,
        scale_factor=scale_factor,
        patch_size=patch_size_lr
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    val_dataset = CustomImageDataset(
        hr_dir=val_hr_data_path,
        lr_dir=val_lr_data_path,
        scale_factor=scale_factor,
        patch_size=patch_size_lr
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_features = 64
    num_blocks = 16

    model = EDSR(scale_factor=scale_factor, num_channels=3, num_features=num_features, num_blocks=num_blocks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # LR Scheduler: уменьшаем каждые 10 эпох
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 50

    # Списки для логирования
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for lr_batch, hr_batch in train_bar:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = None
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for lr_batch, hr_batch in val_bar:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                sr_batch = model(lr_batch)
                loss = criterion(sr_batch, hr_batch)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Лучшая модель сохранена: {best_path} (Val Loss: {avg_val_loss:.6f})")

        # Логирование
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {lr_current:.2e}")

        scheduler.step()

    # Графики для лоссов
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    if val_losses:
        plt.plot(val_losses, label="Val Loss", marker='s')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = os.path.join(checkpoint_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"График лосса сохранен: {loss_plot_path}")


    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Финальная модель сохранена: {final_path}")

if __name__ == "__main__":
    main()