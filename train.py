import h5py
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from models.ResVAE import ResVariationalAutoEncoder

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAT_PATH = "data/preprocessed.theta.mat"
MODEL_SAVE_PATH = "checkpoints/ResVAE.pth"
SEED = 717

INPUT_DIM = 100  # Adjusted input dimension to match your complex data
H_DIM = 2000
Z_DIM = 4
H_LAYERS = [2]

NUM_EPOCHS = 20000
BATCH_SIZE = 1024  # Adjusted batch size
LR_RATE = 1e-4
KL_RATE = 0.1

torch.manual_seed(SEED)


def load_cw(mat_path):
    with h5py.File(mat_path, "r") as mat_file:
        # Access the 'bc_dict' group
        bc_dict_group = mat_file["bc_dict"]
        # Initialize a 3D NumPy array to store theta
        conformalWeldings = np.empty(
            (len(bc_dict_group), 100), dtype=np.float32
        )  # Use float32 instead of complex128

        # Iterate over the fields (e.g., 'Case00_12', 'Case00_13', etc.)
        for i, field_name in enumerate(bc_dict_group):
            case_group = bc_dict_group[field_name]
            # xq = case_group['x'][:]  # Load the 'x' dataset into a structured array
            # yq = case_group['y'][:]  # Load the 'y' dataset into a structured array
            theta = case_group["theta"][
                :
            ]  # Load the 'theta' dataset into a structured array
            theta = np.insert(theta, 100, 2 * np.pi)
            theta = np.diff(theta)  # Use diff between theta to train
            # theta_ma = case_group['theta_ma'][:]  # Load the 'theta_ma' dataset into a structured array
            # theta_ma = np.insert(theta_ma, 0, 0)
            # theta_ma = np.diff(theta_ma)
            # bias = case_group['bias'][:]  # Load the 'bias' dataset into a structured array
            conformalWeldings[i] = np.log(
                1 / theta
            )  # Correspond theta and bias together
    return conformalWeldings


def get_kl_rate(epoch, n=500, m=1000):
    if epoch < m:
        return 0
    else:
        return (epoch % n) / n


def train(is_load=False):
    cw = load_cw(MAT_PATH)
    cw_tensor = torch.tensor(cw).to(DEVICE)

    train_data = TensorDataset(cw_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = ResVariationalAutoEncoder(
        input_dim=INPUT_DIM, h_dim=H_DIM, h_layers=H_LAYERS, z_dim=Z_DIM
    ).to(DEVICE)

    ## Load model
    if is_load:
        model = torch.load(MODEL_SAVE_PATH)

    optimizer = optim.Adam(
        model.parameters(), lr=LR_RATE, weight_decay=1e-5, betas=(0.5, 0.999)
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8000], gamma=0.5)

    model.to(DEVICE)

    loader_size = len(train_loader)
    loss_list_dict = {}

    # Start Training
    model.train()

    for epoch in range(NUM_EPOCHS):
        for i, [data] in enumerate(train_loader):
            data = data.to(DEVICE, dtype=torch.float32).view(data.shape[0], INPUT_DIM)
            x_reconstructed, mu, log_var = model(data)

            # Compute loss
            # kl_rate = get_kl_rate(epoch)
            kl_rate = KL_RATE
            loss_dict = model.loss(x_reconstructed, data, mu, log_var, kl_rate)

            # Backprop
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            scheduler.step()

            # Append losses to the lists
            for k, v in loss_dict.items():
                if k not in loss_list_dict:
                    loss_list_dict[k] = np.zeros(loader_size)
                loss_list_dict[k][i] = v.item()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS} | {', '.join([f'{k}: {v.mean():.4f}' for k, v in loss_list_dict.items()])}"
            )

        if epoch % 1000 == 0:
            torch.save(model, MODEL_SAVE_PATH)
