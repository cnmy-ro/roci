import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import datasets


# ---
# Config
LATENT_DIM = 2
HIDDEN_DIM = 512
BATCH_SIZE = 512
ITERS = 3000
BETA = 0.01
DEVICE = 'cuda'
SEED = 0


# ---
# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---
# Utils

def sample_data():
    """
    Data generation process. Unknown to the model.
    """
    X, _ = datasets.make_swiss_roll(n_samples=BATCH_SIZE, noise=0.1)
    data_sample = np.stack([X[:, 0], X[:, 2]], axis=1)
    data_sample = (data_sample - data_sample.min()) / (data_sample.max() - data_sample.min())
    data_sample = data_sample * 2. - 1.
    data_sample = torch.tensor(data_sample, dtype=torch.float, device=DEVICE)
    return data_sample

def sample_model(model):
    latents = torch.randn((BATCH_SIZE, LATENT_DIM), device=DEVICE)
    model_sample = model.decode(latents)
    return model_sample

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM * 2))        
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
            nn.Linear(HIDDEN_DIM, 2))
    def encode(self, x):
        z = self.encoder(x)
        z_mean, z_logvar = z[:, :LATENT_DIM], z[:, LATENT_DIM:]
        return z_mean, z_logvar
    def decode(self, z):
        x = self.decoder(z)
        return x

def kl_loss(z_mean, z_logvar):
    loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
    loss = torch.mean(loss, dim=0)
    return loss


# ---
# Main function
def main():

    # Models and optimizers
    model = VAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Visualization objects
    losses_input_recon, losses_latent_prior = [], []
    data_sample = sample_data().cpu().numpy()
    model_sample = sample_model(model).detach().cpu().numpy()
    fig, ax = plt.subplots()
    data_sample_plot = ax.plot(data_sample[:, 0], data_sample[:, 1], c='tab:blue', marker='.', ls='', label='Data')[0]
    model_sample_plot = ax.plot(model_sample[:, 0], model_sample[:, 1], c='tab:red', marker='.', ls='', label='Model')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Samples")
    fig.legend(); fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    for it in range(ITERS):

        # Update model
        opt.zero_grad()
        data_sample = sample_data()
        latents_mean, latents_logvar = model.encode(data_sample)
        latents = latents_mean + (latents_logvar * 0.5).exp() * torch.randn((BATCH_SIZE, LATENT_DIM), device=DEVICE)  # Reparam trick
        recon = model.decode(latents)
        loss_input_recon = F.mse_loss(recon, data_sample)
        loss_latent_prior = kl_loss(latents_mean, latents_logvar)
        loss = loss_input_recon + BETA * loss_latent_prior
        loss.backward()
        opt.step()

        # Sample and viz
        losses_input_recon.append(loss_input_recon.detach().cpu().numpy()); losses_latent_prior.append(loss_latent_prior.detach().cpu().numpy())
        if it % 10 == 0:
            data_sample = sample_data().cpu().numpy()
            data_sample_plot.set_xdata(data_sample[:, 0]); data_sample_plot.set_ydata(data_sample[:, 1])
            model_sample = sample_model(model).detach().cpu().numpy()
            model_sample_plot.set_xdata(model_sample[:, 0]); model_sample_plot.set_ydata(model_sample[:, 1])
            fig.canvas.draw(); fig.canvas.flush_events()

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(losses_input_recon, label='Recon'); ax.plot(losses_latent_prior, label='Prior')
    ax.set_title("Training loss"); ax.legend(); plt.show()


# ---
# Run
if __name__ == '__main__':
    main()