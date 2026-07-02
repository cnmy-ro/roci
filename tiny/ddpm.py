"""
Demo of a Denoising Diffusion Probabilistic Model (Ho et al. NeurIPS 2020) on 2D swiss-roll toy dataset.
"""


import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---
# Reproducibility
np.random.seed(0)
torch.manual_seed(0)


# ---
# Config
HIDDEN_DIM = 512
NUM_DIFFUSION_STEPS = 1000
BETA_T = 1e-4
BATCH_SIZE = 512
NUM_ITERS = 10000
DEVICE = 'cuda'


# ---
# Precompute coeff schedules
beta_schedule = torch.linspace(1e-6, BETA_T, NUM_DIFFUSION_STEPS, device=DEVICE) # Linear schedule
sigma_schedule = torch.sqrt(beta_schedule)
alpha_schedule = 1 - beta_schedule
alpha_bar_schedule = torch.empty_like(alpha_schedule)
alpha_bar_schedule[0] = alpha_schedule[0]
for t in range(1, NUM_DIFFUSION_STEPS):
    alpha_bar_schedule[t] = torch.prod(alpha_schedule[:t])


# ---
# Utils

class NoiseModel(torch.nn.Module):
    def __init__(self):        
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), 
            nn.Linear(HIDDEN_DIM, 2)
            ).to(DEVICE)
    
    def forward(self, x, t):
        t = t / NUM_DIFFUSION_STEPS
        t = t * 2. - 1.
        input = torch.cat((x, t), dim=1)
        return self.model(input)

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

@torch.no_grad()
def sample_model(noise_model):
     
    model_sample = torch.randn((BATCH_SIZE, 2), device=DEVICE)        

    for t in range(NUM_DIFFUSION_STEPS - 1, 0, -1):

        z = torch.randn_like(model_sample) if t > 0 else torch.zeros_like(model_sample)
        t_batch = torch.full((BATCH_SIZE, 1), t, device=DEVICE, dtype=torch.long)
        sigma_t, alpha_t, alpha_bar_t = sigma_schedule[t_batch], alpha_schedule[t_batch], alpha_bar_schedule[t_batch]
        noise_pred = noise_model(model_sample, t_batch)
        model_sample = (1 / torch.sqrt(alpha_t)) * (model_sample - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) + \
                        sigma_t * z

    return model_sample

@torch.no_grad()
def show_reverse_diffusion(noise_model):
    
    model_sample = torch.randn((BATCH_SIZE, 2), device=DEVICE)

    fig, ax = plt.subplots()
    model_sample_plot = ax.plot(model_sample.cpu().numpy()[:, 0], model_sample.cpu().numpy()[:, 1], c='tab:red', marker='.', ls='')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Reverse diffusion process")        
    fig.tight_layout(); plt.ion(); plt.show()    

    for t in range(NUM_DIFFUSION_STEPS - 1, 0, -1):
        
        z = torch.randn_like(model_sample) if t > 0 else torch.zeros_like(model_sample)
        t_batch = torch.ones((BATCH_SIZE, 1), device=DEVICE, dtype=torch.long) * t            
        sigma_t, alpha_t, alpha_bar_t = sigma_schedule[t_batch], alpha_schedule[t_batch], alpha_bar_schedule[t_batch]
        noise_pred = noise_model(model_sample, t_batch)
        model_sample = (1 / torch.sqrt(alpha_t)) * (model_sample - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) + \
                        sigma_t * z

        model_sample_plot.set_xdata(model_sample.cpu().numpy()[:, 0])
        model_sample_plot.set_ydata(model_sample.cpu().numpy()[:, 1])
        fig.canvas.draw(); fig.canvas.flush_events()
    
    plt.ioff()

def criterion(data_sample, t_batch, noise_model):
    alpha_bar_t = alpha_bar_schedule[t_batch]
    std_noise = torch.randn_like(data_sample)
    noise_pred = noise_model(torch.sqrt(alpha_bar_t) * data_sample + torch.sqrt(1 - alpha_bar_t) * std_noise, t_batch)
    loss = torch.sum((std_noise - noise_pred) ** 2)
    return loss


# ---
# Main function
def main():

    # Model and optimizer
    noise_model = NoiseModel()
    opt = torch.optim.Adam(noise_model.parameters(), lr=0.0001)

    # Visualization objects
    losses = []
    data_sample = sample_data().cpu().numpy()
    model_sample = sample_model(noise_model).cpu().numpy()
    fig, ax = plt.subplots()
    data_sample_plot = ax.scatter(data_sample[:, 0], data_sample[:, 1], c='tab:blue', marker='.', label='Data')
    model_sample_plot = ax.plot(model_sample[:, 0], model_sample[:, 1], c='tab:red', marker='.', ls='', label='Model')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Samples")
    fig.legend(); fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    for it in tqdm(range(NUM_ITERS)):
    
        # Update noise model
        opt.zero_grad()
        data_sample = sample_data()
        t_batch = torch.randint(0, NUM_DIFFUSION_STEPS, (BATCH_SIZE, 1), device=DEVICE, dtype=torch.long)
        loss = criterion(data_sample, t_batch, noise_model)
        loss.backward()
        opt.step()
        
        # Sample and viz
        losses.append(loss.detach().cpu().numpy())
        if it % 1000 == 0:
            model_sample = sample_model(noise_model).cpu().numpy()
            model_sample_plot.set_xdata(model_sample[:, 0])
            model_sample_plot.set_ydata(model_sample[:, 1])
            fig.canvas.draw(); fig.canvas.flush_events() 

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(losses); ax.set_title("Training loss"); plt.show()
    show_reverse_diffusion(noise_model)


# ---
# Run
if __name__ == '__main__':
    main()