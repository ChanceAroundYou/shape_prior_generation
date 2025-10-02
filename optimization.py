from copy import deepcopy
from functools import partial

import numpy as np
import optuna
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from models.BPResVAE import BPResVAE
from train import setup_seeds
from utils.load_bp import load_from_dir
from utils.visualize import contour_to_scatter, make_grid

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "data/img/BW fish"
TB_PATH = "runs/BPResVAE_fish"
TB_COMMENT = ""
MODEL_LOAD_PATH = ""
SEED = 717
DTYPE = torch.float64

INPUT_DIM = 200  # Adjusted input dimension to match your complex data
H_DIM = 1000
Z_DIM = 2
H_LAYERS = [2, 4]

NUM_EPOCHS = 80000
BATCH_SIZE = 32  # Adjusted batch size
LR_RATE = 6e-5  # smaller learning rate
KL_RATE = 0.75

input_np = load_from_dir(IMAGE_DIR, 100, 25)
input_tensor = torch.tensor(input_np).to(dtype=DTYPE)


def save_model_result(model, path):
    n = input_tensor.shape[0]
    generated_contour, _, _ = model(
        input_tensor.to(DEVICE, dtype=torch.float64).view(n, INPUT_DIM), 0
    )
    generated_contour = generated_contour.cpu().detach()
    generated_contour = model.reconstruct(generated_contour).numpy()

    ground_truth_contour = input_tensor.cpu().numpy()

    generated_images = [
        contour_to_scatter(contour, color="b", figsize=(2, 2), pointsize=2)
        for contour in generated_contour
    ]
    ground_truth_images = [
        contour_to_scatter(contour, color="r", figsize=(2, 2), pointsize=2)
        for contour in ground_truth_contour
    ]
    ground_truth_grid = make_grid(
        ground_truth_images, rows=5, title="Ground Truth", figsize=(10, 11)
    )
    generated_grid = make_grid(
        generated_images, rows=5, title="Encode-decode Result", figsize=(10, 11)
    )
    combined_grid = make_grid(
        [ground_truth_grid, generated_grid], rows=1, figsize=(20, 11)
    )
    plt.imsave(f"{path}/result.png", combined_grid)


def objective(trial, input_tensor, input_dim, device, study_name):
    def _get_value_from_loss(loss_dict, rate=0.25):
        return loss_dict["recon_loss"].item() + rate * loss_dict["kl_loss"].item()

    # Define hyperparameters to optimize
    z_dim = trial.suggest_int("z_dim", 1, 5)
    h_dim = trial.suggest_int("h_dim", 500, 800)
    h_layers = [
        trial.suggest_int("h_layers_1", 1, 3),
        trial.suggest_int("h_layers_2", 1, 3),
        trial.suggest_int("h_layers_3", 1, 3),
    ]
    lr_rate = trial.suggest_float("lr_rate", 1e-5, 2e-3, log=True)
    kl_rate = trial.suggest_float("kl_rate", 0.1, 0.5)
    num_epochs = trial.suggest_int("num_epochs", 8000, 12000)

    tb_path = f"runs/{study_name}/{trial.number}"

    # Create and train model with suggested parameters
    model = BPResVAE(
        input_dim=input_dim,
        hidden_dim=h_dim,
        hidden_layers=h_layers,
        latent_dim=z_dim,
        device=device,
    )

    # 初始化模型
    model = model.to(device).to(DTYPE)
    model.initial()

    # 设置优化器
    optimizer = optim.Adam(
        model.parameters(), lr=lr_rate, weight_decay=1e-5, betas=(0.5, 0.999)
    )

    # TensorBoard
    writer = SummaryWriter(tb_path)
    model_save_path = f"{tb_path}/model.pth"

    input_size = input_tensor.shape[0]
    input_data = input_tensor.to(device, dtype=DTYPE).view(input_size, input_dim)

    best_loss = None

    for epoch in range(num_epochs):
        x_reconstructed, mu, logvar = model(input_data)
        loss_dict = model.loss(x_reconstructed, input_data, mu, logvar, kl_rate)
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        for k in loss_dict.keys():
            writer.add_scalars(k, {"train": loss_dict[k].item()}, epoch)

        # 保存最佳模型
        if best_loss is None or best_loss["loss"].item() > loss_dict["loss"].item():
            best_loss = loss_dict
            print(
                f"Best epoch {epoch}/{num_epochs} | "
                + ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            )
            best_params = deepcopy(model.state_dict())

        # 定期打印
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}/{num_epochs} | "
                + ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            )

        intermediate_value = _get_value_from_loss(loss_dict)
        trial.report(intermediate_value, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    # 保存最佳模型
    model.load_state_dict(best_params)
    torch.save(best_params, model_save_path)
    writer.close()

    save_model_result(model, tb_path)
    # Return the final loss as the objective value
    value = _get_value_from_loss(best_loss)
    return value


# Create study

setup_seeds(SEED)
study_name = "bpvae_op3"
db = "postgresql://postgres:postgres@192.168.1.6/postgres"
study = optuna.create_study(
    study_name=study_name,
    storage=db,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=2000, interval_steps=50
    ),
)

# Run optimization
objective_with_fixed_params = partial(
    objective,
    input_tensor=input_tensor,
    input_dim=INPUT_DIM,
    device=DEVICE,
    study_name=study_name,
)
study.optimize(objective_with_fixed_params, n_trials=100)

# Print results
print("Best trial:")
print("  Value: ", study.best_trial.value)
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
