import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import datasets


# ---
# Reproducibility
np.random.seed(0)
torch.manual_seed(0)


# ---
# Config
LATENT_DIM = 2
HIDDEN_DIM = 512
BATCH_SIZE = 512
LOSS_TYPE = 'nsgan'
ITERS = 1000
DIS_ITERS = 10
DEVICE = 'cuda'


# ---
# Utils

def sample_data():
    """
    Data generation process. Unknown to the model.
    """
    X, _ = datasets.make_swiss_roll(n_samples=BATCH_SIZE, noise=0.25)
    data_sample = np.stack([X[:, 0], X[:, 2]], axis=1)
    data_sample = (data_sample - data_sample.min()) / (data_sample.max() - data_sample.min())
    data_sample = data_sample * 2. - 1.
    data_sample = torch.tensor(data_sample, dtype=torch.float, device=DEVICE)
    return data_sample

def sample_model(generator):
    latent = torch.randn((BATCH_SIZE, LATENT_DIM), device=DEVICE)
    model_sample = generator(latent)
    return model_sample

def compute_disriminator_landscape(discriminator):
    xx, yy = np.meshgrid(np.arange(-1.2, 1.2, 0.01), np.arange(-1.2, 1.2, 0.01))
    xx_t = torch.tensor(xx.flatten(), dtype=torch.float, device=DEVICE)
    yy_t = torch.tensor(yy.flatten(), dtype=torch.float, device=DEVICE)
    with torch.no_grad():
        pred_landscape = discriminator(torch.stack((xx_t, yy_t), dim=1))
        if LOSS_TYPE == 'nsgan': pred_landscape = pred_landscape.sigmoid()
        pred_landscape = pred_landscape.cpu().numpy().reshape(xx.shape)
    return xx, yy, pred_landscape

def add_noise(x, iter_counter):
    return x + 0.05 * (1 - iter_counter/ITERS) * torch.randn_like(x)

def criterion(pred, is_real):
    if LOSS_TYPE == 'nsgan':
        if is_real: target = torch.ones_like(pred)
        else:       target = torch.zeros_like(pred)
        loss = F.binary_cross_entropy_with_logits(pred, target)
    elif LOSS_TYPE == 'wgan':
        if is_real:  loss = -torch.mean(pred)
        else:        loss = torch.mean(pred)    
    elif LOSS_TYPE == 'lsgan':
        if is_real:  loss = 0.5 * torch.mean((pred - 1) ** 2)
        else:        loss = 0.5 * torch.mean(pred ** 2)
    return loss


# ---
# Main function
def main():

    # Models and optimizers
    generator = nn.Sequential(
        nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
        nn.Linear(HIDDEN_DIM, 2)
        ).to(DEVICE)
    discriminator = nn.Sequential(
        nn.Linear(2, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
        nn.Linear(HIDDEN_DIM, 1)
        ).to(DEVICE)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    # Visualization objects
    losses_g, losses_d = [], []
    data_sample = sample_data().cpu().numpy()
    model_sample = sample_model(generator).detach().cpu().numpy()
    xx, yy, pred_landscape = compute_disriminator_landscape(discriminator)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, pred_landscape)
    data_sample_plot = ax.plot(data_sample[:, 0], data_sample[:, 1], c='tab:blue', marker='.', ls='', label='Data')[0]
    model_sample_plot = ax.plot(model_sample[:, 0], model_sample[:, 1], c='tab:red', marker='.', ls='', label='Model')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Samples")
    fig.legend(); fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    for it in range(ITERS):

        # Update D
        for p in discriminator.parameters(): p.requires_grad = True
        for _ in range(DIS_ITERS):
            opt_d.zero_grad()
            data_sample = add_noise(sample_data(), it)
            model_sample = add_noise(sample_model(generator).detach(), it)
            loss_d = criterion(discriminator(data_sample), is_real=True) + criterion(discriminator(model_sample), is_real=False)
            loss_d.backward()
            opt_d.step()
        
        # Update G
        for p in discriminator.parameters(): p.requires_grad = False
        opt_g.zero_grad()
        model_sample = sample_model(generator)
        loss_g = criterion(discriminator(model_sample), is_real=True)
        loss_g.backward()
        opt_g.step()

        # Sample and viz
        losses_g.append(loss_g.detach().cpu().numpy()); losses_d.append(loss_d.detach().cpu().numpy())
        if it % 10 == 0:
            xx, yy, pred_landscape = compute_disriminator_landscape(discriminator)
            ax.contourf(xx, yy, pred_landscape)
            data_sample = sample_data().cpu().numpy()
            data_sample_plot.set_xdata(data_sample[:, 0]); data_sample_plot.set_ydata(data_sample[:, 1])
            model_sample = sample_model(generator).detach().cpu().numpy()
            model_sample_plot.set_xdata(model_sample[:, 0]); model_sample_plot.set_ydata(model_sample[:, 1])
            fig.canvas.draw(); fig.canvas.flush_events()

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(losses_g, label='G'); ax.plot(losses_d, label='D')
    ax.set_title("Training loss"); ax.legend(); plt.show()


# ---
# Run
if __name__ == '__main__':
    main()