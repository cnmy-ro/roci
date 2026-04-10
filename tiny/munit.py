"""
UNIT/MUNIT, but tiny version for modelling 2D data distributions.

Given:
- Samples from arbitrary distribs (domains) A and B

UNIT postulates the existence of the following:
- A shared feature (content) distrib from which both domain distribs are hypothesized to spring from

MUNIT postulates the existence of one additional thing:
- Domain-specific latent (style) distribs which encode the multiple modes of the content-conditioned domain distribs. 
Style latent code represents features of one distrib chich cannot be explained by the other distrib (via the content)
"""

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



# ---
# Config

MULTIMODAL = True  # If `True`, data distribs are multimodal and MUNIT is used to model them. Otherwise, distribs are unimodal and UNIT is used.
HIDDEN_DIM = 512
NUM_HIDDEN_LAYERS = 6
BATCH_SIZE = 100
CONTENT_DIM = 2  # 2 is weak inductive bias on content; 1 is strong inductive bias where content is assumed to lie on a circle
STYLE_DIM = 2
STYLE_PRIOR = 'gaussian'  # gaussian, uniform, bernoulli
ITERS = 10000
DIS_ITERS = 1
LAMBDA_INPUT_RECON = 10
LAMBDA_CYCLE = 10
LAMBDA_CONTENT_RECON = 1
LAMBDA_STYLE_RECON = 1
SEED = 2
DEVICE = 'cuda'



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
    
    data_noise_level = 0.01
    scale_factor = 0.2
    margin = np.pi/6
    
    def sample_content():
        # Unit circle
        angles = np.random.uniform(0+margin, np.pi-margin, size=int(BATCH_SIZE))
        content = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return content
    
    def fn_a(content):        
        a = content.copy()
        if not MULTIMODAL:
            a = scale_factor * 2 * a
        if MULTIMODAL:  # Factors of variation of distrib A not explained by the content (and hence by distrib B)
            mask = np.random.choice([False, True], content.shape[0], p=[0.4, 0.6])
            a[mask, :] = scale_factor * 2 * a[mask, :]
            a[np.logical_not(mask), :] = scale_factor * 6 * a[np.logical_not(mask), :]
        noise = np.random.normal(0, data_noise_level, size=(BATCH_SIZE, 2))
        return a + noise
    
    def fn_b(content):
        a = content.copy()
        if not MULTIMODAL:
            a = scale_factor * -2 * a
        if MULTIMODAL:  # Factors of variation of distrib A not explained by the content (and hence by distrib B)
            mask = np.random.choice([False, True], content.shape[0], p=[0.9, 0.1])
            a[mask, :] = scale_factor * -2 * a[mask, :]
            a[np.logical_not(mask), :] = scale_factor * -6 * a[np.logical_not(mask), :]
        noise = np.random.normal(0, data_noise_level, size=(BATCH_SIZE, 2))
        return a + noise
    
    content = sample_content()
    a = fn_a(content)
    b = fn_b(content)

    np.random.shuffle(b)  # To break the A-B pairing
    
    a = torch.tensor(a, dtype=torch.float, device=DEVICE)
    b = torch.tensor(b, dtype=torch.float, device=DEVICE)
    content = torch.tensor(content, dtype=torch.float, device=DEVICE)
    return a, b, content


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = [nn.Linear(2, HIDDEN_DIM), nn.LeakyReLU(0.1)]
        for _ in range(NUM_HIDDEN_LAYERS): self.content_encoder += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LeakyReLU(0.1)]
        self.content_encoder += [nn.Linear(HIDDEN_DIM, CONTENT_DIM)]
        self.content_encoder = nn.Sequential(*self.content_encoder)
        if MULTIMODAL:
            self.style_encoder = [nn.Linear(2, HIDDEN_DIM), nn.LeakyReLU(0.1)]
            for _ in range(NUM_HIDDEN_LAYERS): self.style_encoder += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LeakyReLU(0.1)]
            self.style_encoder += [nn.Linear(HIDDEN_DIM, STYLE_DIM)]
            self.style_encoder = nn.Sequential(*self.style_encoder)
            decoder_input_dims = CONTENT_DIM + STYLE_DIM
        else:
            decoder_input_dims = CONTENT_DIM
        self.decoder = [nn.Linear(decoder_input_dims, HIDDEN_DIM), nn.LeakyReLU(0.1)]
        for _ in range(NUM_HIDDEN_LAYERS): self.decoder += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LeakyReLU(0.1)]
        self.decoder += [nn.Linear(HIDDEN_DIM, 2)]
        self.decoder = nn.Sequential(*self.decoder)
    def encode(self, x):
        content = self.content_encoder(x)
        if MULTIMODAL: style = self.style_encoder(x)
        else:          style = None
        return content, style
    def decode(self, content, style=None):
        if MULTIMODAL: return self.decoder(torch.cat([content, style], dim=1))
        else:          return self.decoder(content)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = [nn.Linear(2, HIDDEN_DIM), nn.LeakyReLU(0.1)]
        for _ in range(NUM_HIDDEN_LAYERS+2): self.model += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LeakyReLU(0.1)]
        self.model += [nn.Linear(HIDDEN_DIM, 1)]
        self.model = nn.Sequential(*self.model)
    def forward(self, x):
        return self.model(x)


def init_networks():
    networks = {}
    networks['autoenc_a'] = AutoEncoder().to(DEVICE)
    networks['autoenc_b'] = AutoEncoder().to(DEVICE)
    networks['dis_a'] = Discriminator().to(DEVICE)
    networks['dis_b'] = Discriminator().to(DEVICE)
    return networks


def init_optimizers(networks):
    params_autoenc = itertools.chain( networks['autoenc_a'].parameters(), networks['autoenc_b'].parameters())
    params_dis = itertools.chain(networks['dis_a'].parameters(), networks['dis_b'].parameters())
    optimizers = {}
    optimizers['autoenc'] = torch.optim.Adam(params_autoenc, lr=0.0001)
    optimizers['dis'] = torch.optim.Adam(params_dis, lr=0.0001) 
    return optimizers


def sample_style_from_prior():
    if STYLE_PRIOR == 'bernoulli':  style = torch.randint(0, 2, (BATCH_SIZE, STYLE_DIM), dtype=torch.float, device=DEVICE) * 2 - 1
    elif STYLE_PRIOR == 'uniform':  style = torch.rand((BATCH_SIZE, STYLE_DIM), dtype=torch.float, device=DEVICE) * 2 - 1
    elif STYLE_PRIOR == 'gaussian': style = torch.randn((BATCH_SIZE, STYLE_DIM), dtype=torch.float, device=DEVICE)
    return style


def translate(networks, data_sample, to_domain='b'):
    if MULTIMODAL: style = sample_style_from_prior()
    else:          style = None        

    if to_domain == 'b':
        content, _ = networks['autoenc_a'].encode(data_sample)
        output = networks['autoenc_b'].decode(content, style)
    elif to_domain == 'a':
        content, _ = networks['autoenc_b'].encode(data_sample)
        output = networks['autoenc_a'].decode(content, style)

    return output


def gan_criterion(pred, is_real):
    if is_real: target = torch.ones_like(pred)
    else:       target = torch.zeros_like(pred)
    loss = F.mse_loss(pred, target)
    return loss


def init_mapping_plot(ax, input, output):
    mapping_plots = []
    for i in range(BATCH_SIZE):
        p = ax.plot([input[i, 0], output[i, 0]], [input[i, 1], output[i, 1]], c='black', marker='', ls='-', alpha=0.2)[0]
        mapping_plots.append(p)
    return mapping_plots


def update_mapping_plot(mapping_plots, input, output):
    for i in range(BATCH_SIZE):
        mapping_plots[i].set_xdata([input[i, 0], output[i, 0]])
        mapping_plots[i].set_ydata([input[i, 1], output[i, 1]])



# ---
# Main

def main():

    # Model and optimizers
    networks = init_networks()
    optimizers = init_optimizers(networks)

    # Visualization objects
    losses_traj = {
        'input_recon_a': [], 'input_recon_b': [],
        'cycle_a': [], 'cycle_b': [],
        'gan_autoenc_a': [], 'gan_autoenc_b': [],
        'gan_dis_a': [], 'gan_dis_b': []
        }
    if MULTIMODAL:
        losses_traj.update({'content_recon_a': [], 'content_recon_b': [], 'style_recon_a': [], 'style_recon_b': []})
    a, b, content_orig = sample_data()
    with torch.no_grad():
        ab = translate(networks, a, to_domain='b').detach().cpu().numpy()
        ba = translate(networks, b, to_domain='a').detach().cpu().numpy()    
    a, b = a.cpu().numpy(), b.cpu().numpy()
    content_orig = content_orig.cpu().detach().numpy()
    style_a_prior = sample_style_from_prior()
    style_b_prior = sample_style_from_prior()
    style_a_prior, style_b_prior = style_a_prior.cpu().detach().numpy(), style_b_prior.cpu().detach().numpy()

    if MULTIMODAL and STYLE_DIM == 2: fig, axs = plt.subplots(1, 5, figsize=(20, 4)) 
    else:                             fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    a_plot = axs[0].plot(a[:, 0], a[:, 1], c='tab:blue', marker='.', ls='', label='Data: A')[0]
    ab_plot = axs[0].plot(ab[:, 0], ab[:, 1], c='magenta', marker='.', ls='', label='Model: A -> B')[0]
    ab_mapping_plots = init_mapping_plot(axs[0], a, ab)
    b_plot = axs[1].plot(b[:, 0], b[:, 1], c='tab:red', marker='.', ls='', label='Data: B')[0]
    ba_plot = axs[1].plot(ba[:, 0], ba[:, 1], c='cyan', marker='.', ls='', label='Model: B -> A')[0]
    ba_mapping_plots = init_mapping_plot(axs[1], b, ba)
    content_orig_plot = axs[2].plot(content_orig[:, 0], content_orig[:, 1], c='tab:brown', marker='.', ls='', label='Hidden true content')[0]
    content_a_plot = axs[2].plot(content_orig[:, 0], content_orig[:, 1], c='tab:orange', marker='.', ls='', label='Learned pragmatic content (A)')[0]
    content_b_plot = axs[2].plot(content_orig[:, 0], content_orig[:, 1], c='tab:olive', marker='.', ls='', label='Learned pragmatic content (B)')[0]
    if MULTIMODAL and STYLE_DIM == 2:
        style_a_prior_plot = axs[3].plot(style_a_prior[:, 0], style_a_prior[:, 1], c='tab:blue', marker='.', ls='', label='Prior style (A)')[0]
        style_a_plot = axs[3].plot(style_a_prior[:, 0], style_a_prior[:, 1], c='cyan', marker='.', ls='', label='Learned style (A)')[0]
        style_b_prior_plot = axs[4].plot(style_b_prior[:, 0], style_b_prior[:, 1], c='tab:red', marker='.', ls='', label='Prior style (B)')[0]
        style_b_plot = axs[4].plot(style_b_prior[:, 0], style_b_prior[:, 1], c='magenta', marker='.', ls='', label='Learned style (B)')[0]
    for ax in axs.ravel(): ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.legend(loc='upper right')
    fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    loss = {k: None for k in losses_traj.keys()}
    for it in range(ITERS):

        # Update discriminators
        for p in networks['dis_a'].parameters(): p.requires_grad = True
        for p in networks['dis_b'].parameters(): p.requires_grad = True
        for _ in range(DIS_ITERS):
            optimizers['dis'].zero_grad()
            a, b, _ = sample_data()
            ab = translate(networks, a, to_domain='b').detach()
            ba = translate(networks, b, to_domain='a').detach()
            loss['gan_dis_a'] = gan_criterion(networks['dis_a'](a), is_real=True) + gan_criterion(networks['dis_a'](ba), is_real=False)
            loss['gan_dis_b'] = gan_criterion(networks['dis_b'](b), is_real=True) + gan_criterion(networks['dis_b'](ab), is_real=False)
            loss_dis = loss['gan_dis_a'] + loss['gan_dis_b']
            loss_dis.backward()
            optimizers['dis'].step()
        
        # Update autoencoders
        for p in networks['dis_a'].parameters(): p.requires_grad = False
        for p in networks['dis_b'].parameters(): p.requires_grad = False
        optimizers['autoenc'].zero_grad()
        
        a, b, content_orig = sample_data()
        content_a, style_a = networks['autoenc_a'].encode(a)
        content_b, style_b = networks['autoenc_b'].encode(b)
        aa = networks['autoenc_a'].decode(content_a, style_a)
        bb = networks['autoenc_b'].decode(content_b, style_b)
        if MULTIMODAL:
            rand_style_a = sample_style_from_prior()
            rand_style_b = sample_style_from_prior()
        else:
            rand_style_a = rand_style_b = None
        ab = networks['autoenc_b'].decode(content_a, rand_style_b)
        ba = networks['autoenc_a'].decode(content_b, rand_style_a)
        content_ab, style_ab = networks['autoenc_b'].encode(ab)
        content_ba, style_ba = networks['autoenc_a'].encode(ba)
        aba = networks['autoenc_a'].decode(content_ab, style_a)
        bab = networks['autoenc_b'].decode(content_ba, style_b)

        loss['input_recon_a'] = F.mse_loss(aa, a)
        loss['input_recon_b'] = F.mse_loss(bb, b)
        loss['cycle_a'] = F.mse_loss(aba, a)
        loss['cycle_b'] = F.mse_loss(bab, b)
        loss['gan_autoenc_a'] = gan_criterion(networks['dis_a'](ba), is_real=True)
        loss['gan_autoenc_b'] = gan_criterion(networks['dis_b'](ab), is_real=True)
        loss_autoenc = loss['gan_autoenc_a'] + loss['gan_autoenc_b']                        + \
                       LAMBDA_INPUT_RECON * (loss['input_recon_a'] + loss['input_recon_a']) + \
                       LAMBDA_CYCLE * (loss['cycle_a'] + loss['cycle_b'])
        if MULTIMODAL:
            loss['content_recon_a'] = F.mse_loss(content_ab, content_a.detach())
            loss['content_recon_b'] = F.mse_loss(content_ba, content_b.detach())
            loss['style_recon_a'] = F.mse_loss(style_ba, rand_style_a)
            loss['style_recon_b'] = F.mse_loss(style_ab, rand_style_b)
            loss_autoenc += LAMBDA_CONTENT_RECON * (loss['content_recon_a'] + loss['content_recon_b']) + \
                            LAMBDA_STYLE_RECON * (loss['style_recon_a'] + loss['style_recon_b'])
        loss_autoenc.backward()
        optimizers['autoenc'].step()

        # Track and viz
        for k in losses_traj.keys(): losses_traj[k].append(loss[k].detach().cpu().numpy())
        if it % 10 == 0:
            a, b = a.cpu().detach().numpy(), b.cpu().detach().numpy()
            ab, ba = ab.cpu().detach().numpy(), ba.cpu().detach().numpy()
            content_orig = content_orig.cpu().detach().numpy()
            content_a, content_b = content_a.cpu().detach().numpy(), content_b.cpu().detach().numpy()
            if CONTENT_DIM == 1:
                content_a = np.stack([np.cos(np.pi*content_a).squeeze(), np.sin(np.pi*content_a).squeeze()], axis=1)
                content_b = np.stack([np.cos(np.pi*content_b).squeeze(), np.sin(np.pi*content_b).squeeze()], axis=1)
            a_plot.set_xdata(a[:, 0]); a_plot.set_ydata(a[:, 1])            
            ab_plot.set_xdata(ab[:, 0]); ab_plot.set_ydata(ab[:, 1])
            update_mapping_plot(ab_mapping_plots, a, ab)
            b_plot.set_xdata(b[:, 0]); b_plot.set_ydata(b[:, 1])
            ba_plot.set_xdata(ba[:, 0]); ba_plot.set_ydata(ba[:, 1])
            update_mapping_plot(ba_mapping_plots, b, ba)
            content_orig_plot.set_xdata(content_orig[:, 0]); content_orig_plot.set_ydata(content_orig[:, 1])
            content_a_plot.set_xdata(content_a[:, 0]); content_a_plot.set_ydata(content_a[:, 1])
            content_b_plot.set_xdata(content_b[:, 0]); content_b_plot.set_ydata(content_b[:, 1])
            if MULTIMODAL and STYLE_DIM == 2:
                rand_style_a, rand_style_b = rand_style_a.cpu().detach().numpy(), rand_style_b.cpu().detach().numpy()
                style_a, style_b = style_a.cpu().detach().numpy(), style_b.cpu().detach().numpy()
                style_a_prior_plot.set_xdata(rand_style_a[:, 0]); style_a_prior_plot.set_ydata(rand_style_a[:, 1])
                style_a_plot.set_xdata(style_a[:, 0]); style_a_plot.set_ydata(style_a[:, 1])
                style_b_prior_plot.set_xdata(rand_style_b[:, 0]); style_b_prior_plot.set_ydata(rand_style_b[:, 1])
                style_b_plot.set_xdata(style_b[:, 0]); style_b_plot.set_ydata(style_b[:, 1])
            fig.canvas.draw(); fig.canvas.flush_events()

    plt.ioff()
    fig, axs = plt.subplots(6, 2, figsize=(8, 14))
    axs[0][0].plot(losses_traj['input_recon_a'], label='input_recon_a');     axs[0][1].plot(losses_traj['input_recon_b'], label='input_recon_b')
    axs[1][0].plot(losses_traj['cycle_a'], label='cycle_a');                 axs[1][1].plot(losses_traj['cycle_b'], label='cycle_b')
    axs[2][0].plot(losses_traj['content_recon_a'], label='content_recon_a'); axs[2][1].plot(losses_traj['content_recon_b'], label='content_recon_b')
    axs[3][0].plot(losses_traj['style_recon_a'], label='style_recon_a');     axs[3][1].plot(losses_traj['style_recon_b'], label='style_recon_b')
    axs[4][0].plot(losses_traj['gan_autoenc_a'], label='gan_autoenc_a');     axs[4][1].plot(losses_traj['gan_autoenc_b'], label='gan_autoenc_b')
    axs[5][0].plot(losses_traj['gan_dis_a'], label='gan_dis_a');             axs[5][1].plot(losses_traj['gan_dis_b'], label='gan_dis_b')
    [ax.legend() for ax in axs.ravel()]
    fig.suptitle("Training loss"); fig.tight_layout();  plt.show()



# ---
# Run

if __name__ == '__main__':
    main()