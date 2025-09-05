import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128, inference_mode=False):
        super(VAE, self).__init__()

        self.inference_mode = inference_mode

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),          # 1x64x512  -> 32x32x256
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),                      # 32x32x256 -> 64x16x128
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),                     # 64x16x128 -> 128x8x64
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),                    # 128x8x64  -> 256x4x32
            nn.ELU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),                    # 256x4x32  -> 512x2x16
            nn.ELU(),
        )

        # Compute the mean and log variance for the latent space
        self.fc_mu = nn.Linear(512 * 2 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 16, latent_dim)

        # Decoder network
        self.decoder_fc = nn.Linear(latent_dim, 512 * 2 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),           # 512x2x16  -> 256x4x32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),           # 256x4x32  -> 128x8x64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),            # 128x8x64  -> 64x16x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),             # 64x16x128 -> 32x32x256
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # 32x32x256 -> 1x64x512
            nn.Sigmoid()                                                                # Using Sigmoid to constrain output between 0 and 1
        )

    def encode(self, x):
        # Pass input through encoder
        x = self.encoder(x)
        # Flatten the output from the convolutional layers
        x = torch.flatten(x, start_dim=1)
        # Compute the mean and log variance for the latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * logvar)
        # Sample from a normal distribution
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        # Reparameterize to obtain latent vector z
        z = mu + eps * std
        return z

    def decode(self, z):
        # Pass latent vector through intermediate fully connected layers
        x = self.decoder_fc(z)
        # Reshape to 4D tensor
        x = x.view(-1, 512, 2, 16)
        # Pass through the decoder network
        x = self.decoder(x)
        return x

    def forward(self, x):
        # Encode input to latent space
        mu, logvar = self.encode(x)
        # Reparameterize to sample from the latent space
        z = self.reparameterize(mu, logvar)
        # Decode the latent vector to reconstruct the image
        x_logit = self.decode(z)
        
        return x_logit, mu, logvar
    

class VAE_2(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128, inference_mode=False):
        super(VAE_2, self).__init__()

        self.inference_mode = inference_mode

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),          # 1x16x1800  -> 32x8x900
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),                      # 32x8x900 -> 64x4x450
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),                     # 64x4x450 -> 128x2x225
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),                    # 128x2x225 -> 256x1x112
            nn.ELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),                    # 256x1x112 -> 512x1x112
            nn.ELU(),
        )

        # Compute the mean and log variance for the latent space
        self.fc_mu = nn.Linear(512 * 1 * 112, latent_dim)
        self.fc_logvar = nn.Linear(512 * 1 * 112, latent_dim)

        # Decoder network
        self.decoder_fc = nn.Linear(latent_dim, 512 * 1 * 112)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),                        # 512x1x112 -> 256x1x112
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)), # 256x1x112 -> 128x2x225
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),                         # 128x2x225 -> 64x4x450
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),                          # 64x4x450 -> 32x8x900
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),                           # 32x8x900 -> 1x16x1800
            nn.Sigmoid()                                                                # Using Sigmoid to constrain output between 0 and 1
        )

    def encode(self, x):
        # Pass input through encoder
        x = self.encoder(x)
        # Flatten the output from the convolutional layers
        x = torch.flatten(x, start_dim=1)
        # Compute the mean and log variance for the latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * logvar)
        # Sample from a normal distribution
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        # Reparameterize to obtain latent vector z
        z = mu + eps * std
        return z

    def decode(self, z):
        # Pass latent vector through intermediate fully connected layers
        x = self.decoder_fc(z)
        # Reshape to 4D tensor
        x = x.view(-1, 512, 1, 112)
        # Pass through the decoder network
        x = self.decoder(x)
        return x

    def forward(self, x):
        # Encode input to latent space
        mu, logvar = self.encode(x)
        # Reparameterize to sample from the latent space
        z = self.reparameterize(mu, logvar)
        # Decode the latent vector to reconstruct the image
        x_logit = self.decode(z)
        
        return x_logit, mu, logvar