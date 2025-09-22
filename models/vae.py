import torch
import torch.nn as nn
# VAE Encoder

# class VAEEncoder(nn.Module):
#     def __init__(self, in_channels=3, latent_dim=16):
#         super(VAEEncoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),  # 256 -> 128
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 -> 64
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 64 -> 32
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 32 -> 16
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, latent_dim * 2, 4, stride=2, padding=1),  # 16 -> 8
#         )

#     def forward(self, x):
#         h = self.model(x)
#         mu, logvar = h.chunk(2, dim=1)  # [batch, 16, 8, 8]
        
#         return h,mu, logvar

# # VAE Decoder
# class VAEDecoder(nn.Module):
#     def __init__(self, latent_dim=16, out_channels=3):
#         super(VAEDecoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 512, 4, stride=2, padding=1),  # 16 -> 32
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 32 -> 64
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64 -> 128
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 128 -> 256
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, out_channels, kernel_size=3, padding=1),  # 256 -> 256
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.model(z)

# # VAE Model
# class VAE(nn.Module):
#     def __init__(self, latent_dim=16):
#         super(VAE, self).__init__()
#         self.encoder = VAEEncoder(in_channels=3, latent_dim=latent_dim)
#         self.decoder = VAEDecoder(latent_dim=latent_dim, out_channels=3)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         h, mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decoder(z)
#         # print(z.shape)
#         # print(recon.shape)
#         return recon, h, z

# this vae works for 128 image. developed by prajwal bhaiya

# this vae works for 256 image. developed by me 
# class VAEEncoder(nn.Module):
#     def __init__(self, in_channels=3, latent_dim=16):
#         super(VAEEncoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),  # 256 -> 128
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 -> 64
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 64 -> 32
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 32 -> 16
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, latent_dim * 2, 4, stride=2, padding=1),  # 16 -> 8
#         )

#     def forward(self, x):
#         h = self.model(x)
#         mu, logvar = h.chunk(2, dim=1)  # [batch, 16, 8, 8]
        
#         return h,mu, logvar

# # VAE Decoder
# class VAEDecoder(nn.Module):
#     def __init__(self, latent_dim=16, out_channels=3):
#         super(VAEDecoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 512, 4, stride=2, padding=1),  # 16 -> 32
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 32 -> 64
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64 -> 128
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 128 -> 256
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 256 -> 256
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.model(z)

# # VAE Model
# class VAE(nn.Module):
#     def __init__(self, latent_dim=16):
#         super(VAE, self).__init__()
#         self.encoder = VAEEncoder(in_channels=3, latent_dim=latent_dim)
#         self.decoder = VAEDecoder(latent_dim=latent_dim, out_channels=3)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         h, mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decoder(z)
#         # print(z.shape)
#         # print(recon.shape)
#         return recon, h, z

# #this vae i used in albedo 4-5 million params
# class VAEEncoder(nn.Module):
#     def __init__(self, in_channels=1, latent_dim=8):
#         super(VAEEncoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),  # 256 -> 128
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, 4, stride=2, padding=1), # 32 -> 16
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1), # 128 -> 64
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 512, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, latent_dim * 2, 4, stride=2, padding=1),  # 64 -> 32 
#         )

#     def forward(self, x):
#         h = self.model(x)
#         mu, logvar = h.chunk(2, dim=1)  # [batch, 16, 32, 32]
        
#         return h,mu, logvar


# # # VAE Decoder
# class VAEDecoder(nn.Module):
#     def __init__(self, latent_dim=8, out_channels=1):
#         super(VAEDecoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 512, 4, stride=2, padding=1),  # 32 -> 64
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64 -> 128
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 64 -> 128
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),  # 128 -> 256
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.model(z)

# # VAE Model
# class VAE(nn.Module):
#     def __init__(self, latent_dim=8):
#         super(VAE, self).__init__()
#         self.encoder = VAEEncoder(in_channels=1, latent_dim=latent_dim)
#         self.decoder = VAEDecoder(latent_dim=latent_dim, out_channels=1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         h, mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decoder(z)
#         # print(z.shape)
#         # print(recon.shape)
#         return recon, h, z

# this vae i am using for experiment shading 12-13 m total_params

# class VAEEncoder(nn.Module):
#     def __init__(self, in_channels=1, latent_dim=8):
#         super(VAEEncoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),  # 256 -> 128
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, 4, stride=2, padding=1), # 32 -> 16
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1), # 128 -> 64
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 512, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 4, stride=2, padding=1), # 128 -> 64
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, latent_dim * 2, 3, stride=1, padding=1),  # 64 -> 32 
#         )

#     def forward(self, x):
#         h = self.model(x)
#         mu, logvar = h.chunk(2, dim=1)  # [batch, 16, 32, 32]
        
#         return h,mu, logvar


# # # VAE Decoder
# class VAEDecoder(nn.Module):
#     def __init__(self, latent_dim=8, out_channels=1):
#         super(VAEDecoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 512, 3, stride=1, padding=1),  # 32 -> 64
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),  
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64 -> 128
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 64 -> 128
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),  # 128 -> 256
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.model(z)

# # VAE Model
# class VAE(nn.Module):
#     def __init__(self, latent_dim=8):
#         super(VAE, self).__init__()
#         self.encoder = VAEEncoder(in_channels=1, latent_dim=latent_dim)
#         self.decoder = VAEDecoder(latent_dim=latent_dim, out_channels=1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         h, mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decoder(z)
#         # print(z.shape)
#         # print(recon.shape)
#         return recon, h, z
    


# class VAEEncoder(nn.Module):
#     def __init__(self, in_channels=1, latent_dim=8):
#         super(VAEEncoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),  # 256 -> 128
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1), # 32 -> 16
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),  
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1), # 32 -> 16
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1), # 128 -> 64
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, stride=1, padding=1), # 32 -> 16
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 512, 4, stride=2, padding=1),  
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, stride=1, padding=1), # 128 -> 64
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, latent_dim * 2, 3, stride=1, padding=1),  # 64 -> 32 
#         )

#     def forward(self, x):
#         h = self.model(x)
#         mu, logvar = h.chunk(2, dim=1)  # [batch, 16, 32, 32]
        
#         return h,mu, logvar


# # # VAE Decoder
# class VAEDecoder(nn.Module):
#     def __init__(self, latent_dim=8, out_channels=1):
#         super(VAEDecoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 512, 3, stride=1, padding=1),  # 32 -> 64
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64 -> 128
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),  
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # 64 -> 128
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1),  # 128 -> 256
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.model(z)

# # VAE Model
# class VAE(nn.Module):
#     def __init__(self, latent_dim=8):
#         super(VAE, self).__init__()
#         self.encoder = VAEEncoder(in_channels=1, latent_dim=latent_dim)
#         self.decoder = VAEDecoder(latent_dim=latent_dim, out_channels=1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         h, mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decoder(z)
#         # print(z.shape)
#         # print(recon.shape)
#         return recon, h, z

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=8):
        super(VAEEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),  # 256 -> 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), # 32 -> 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), # 128 -> 64
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, latent_dim * 2, 3, stride=1, padding=1),  # 64 -> 32 
        )

    def forward(self, x):
        h = self.model(x)
        mu, logvar = h.chunk(2, dim=1)  # [batch, 16, 32, 32]
        
        return h,mu, logvar


# # VAE Decoder
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=8, out_channels=1):
        super(VAEDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 3, stride=1, padding=1),  # 32 -> 64
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),  
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),  
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # 64 -> 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  # 64 -> 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  # 64 -> 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),  # 128 -> 256
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(in_channels=1, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, out_channels=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h, mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        # print(z.shape)
        # print(recon.shape)
        return recon, h, z


if __name__ == '__main__':
    x = torch.randn((1, 1, 256, 384))
    
    model = VAE(latent_dim=8)
    recon, h, z = model(x)

    print("Output shapes:")
    print(f"Reconstructed image: {recon.shape}")
    print(f"Intermediate representation h: {h.shape}")
    print(f"Latent vector z: {z.shape}")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in VAE: {total_params:,}")
