import torch
import torch.nn as nn
import numpy as np
from torchvision.models.efficientnet import efficientnet_b0

# [0, 1], Groupnorm
if False:
    class MLP_Encoder(nn.Module):
        def __init__(self, input_channels, hidden_dim1=256, hidden_dim2=128, latent_dim=32, num_groups=8):
            super(MLP_Encoder, self).__init__()

            # Adjust conv_to_latent layers to match channel dimensions correctly
            self.conv_to_latent = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, hidden_dim2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(hidden_dim2, hidden_dim1, kernel_size=3, stride=2, padding=1, bias=True),  
                nn.GroupNorm(num_groups, hidden_dim1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=2, bias=True)
            )

            self.FC_input = nn.Linear(hidden_dim1, hidden_dim1, bias=True)
            self.gn1 = nn.GroupNorm(num_groups, hidden_dim1)
            
            self.FC_hidden = nn.Linear(hidden_dim1, hidden_dim2, bias=True)
            self.gn2 = nn.GroupNorm(num_groups, hidden_dim2)
            
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            h = self.conv_to_latent(x)
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.gn1(self.FC_input(h)))
            h = self.dropout(h)
            h = self.LeakyReLU(self.gn2(self.FC_hidden(h)))
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar

    class CNN_Encoder(nn.Module):
        def __init__(self, latent_dim, num_groups=8):
            super(CNN_Encoder, self).__init__()
            
            # Convolutional blocks: Each block has three layers, but we keep the same number of max-poolings
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2)
            )
            
            self.conv_to_latent = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 384),
                nn.LeakyReLU(0.2),
                nn.Conv2d(384, latent_dim, kernel_size=2, bias=True)
            )
            
            self.fc_layers = MLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim)

        def forward(self, x):
            x = self.conv_layers(x)
            mean, logvar = self.fc_layers(x)
            return mean, logvar

    class Dual_Encoder(nn.Module):
        def __init__(self, scatshape, latent_dim, num_groups=8):
            super(Dual_Encoder, self).__init__()
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2)
            )
            
            self.conv_to_latent_img = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 384),
                nn.LeakyReLU(0.2),
                nn.Conv2d(384, 384, kernel_size=2, bias=True)
            )
            
            self.conv_to_latent_scat = nn.Sequential(
                nn.Conv2d(scatshape[-3], 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 384),
                nn.LeakyReLU(0.2),
                nn.Conv2d(384, 384, kernel_size=2, bias=True)
            )
            
            self.FC_input = nn.Linear(768, 384, bias=True)
            self.gn1 = nn.GroupNorm(num_groups, 384)
            
            self.FC_hidden = nn.Linear(384, 256, bias=True)
            self.gn2 = nn.GroupNorm(num_groups, 256)
            
            self.FC_mean = nn.Linear(256, latent_dim, bias=True)
            self.FC_var = nn.Linear(256, latent_dim, bias=True)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, img, scat):
            cnn_features = self.cnn_encoder(img)
            img_h = self.conv_to_latent_img(cnn_features)
            scat_h = self.conv_to_latent_scat(scat)
            h = torch.cat((img_h, scat_h), dim=1)
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.gn1(self.FC_input(h)))
            h = self.dropout(h)
            h = self.LeakyReLU(self.gn2(self.FC_hidden(h)))
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar

    class CNN_Decoder(nn.Module):
        def __init__(self, latent_dim, intermediate_dim=256, num_groups=8):
            super(CNN_Decoder, self).__init__()
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, intermediate_dim * 4 * 4, bias=True),
                nn.LeakyReLU(0.2),
            )
            
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(intermediate_dim, intermediate_dim // 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, intermediate_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(intermediate_dim // 2, intermediate_dim // 2, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, intermediate_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(intermediate_dim // 2, intermediate_dim // 2, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, intermediate_dim // 2),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(intermediate_dim // 2, 128, kernel_size=4, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=True),
                nn.Sigmoid()
            )

        def forward(self, z):
            z = self.fc_layers(z)
            z = z.view(z.size(0), -1, 4, 4)
            z = self.deconv_layers(z)
            return z


if True: # [0, 1], Batchnorm, J=4
    class MLP_Encoder(nn.Module):
        def __init__(self, input_channels, hidden_dim1=256, hidden_dim2=128, latent_dim=32):
            super(MLP_Encoder, self).__init__()

            # Adjust conv_to_latent layers to match channel dimensions correctly
            self.conv_to_latent = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1, bias=True),  # Downsampling layer
                nn.BatchNorm2d(hidden_dim2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(hidden_dim2, hidden_dim1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=3, stride=2, padding=1, bias=True),  # Downsampling layer
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=2, bias=True)  # Final layer
            )


            self.FC_input = nn.Linear(hidden_dim1, hidden_dim1, bias=True)
            self.bn1 = nn.BatchNorm1d(hidden_dim1)
            
            self.FC_hidden = nn.Linear(hidden_dim1, hidden_dim2, bias=True)
            self.bn2 = nn.BatchNorm1d(hidden_dim2)
            
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            h = self.conv_to_latent(x)
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.bn1(self.FC_input(h)))
            h = self.dropout(h)
            h = self.LeakyReLU(self.bn2(self.FC_hidden(h)))
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar
        
    class CNN_Encoder(nn.Module):
        def __init__(self, latent_dim):
            super(CNN_Encoder, self).__init__()
            
            # Convolutional blocks: Each block has three layers, but we keep the same number of max-poolings
            self.conv_layers = nn.Sequential(
                # Block 1: Input -> 32 x 64 x 64
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 64 x 64

                # Block 2: 32 x 32 x 32
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 32 x 32

                # Block 3: 64 x 16 x 16
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 16 x 16

                # Block 4: 128 x 8 x 8
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 8 x 8

                # Block 5: 256 x 8 x 8 (No MaxPooling)
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)  # Switched to LeakyReLU
            )
            
            # Additional convolutional layers to reduce dimensions
            self.conv_to_latent = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),  # Output: (hidden_dim2, 4, 4)
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),  # Output: (hidden_dim1, 2, 2)
                nn.BatchNorm2d(384),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.Conv2d(384, latent_dim, kernel_size=2, bias=True)  # Output: (latent_dim, 1, 1)
            )
            
            # Fully connected layers for latent space representation
            self.fc_layers = MLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim)

        def forward(self, x):
            x = self.conv_layers(x)
            mean, logvar = self.fc_layers(x)
            return mean, logvar
        
    class Dual_Encoder(nn.Module):
        def __init__(self, scatshape, latent_dim):
            super(Dual_Encoder, self).__init__()
            self.cnn_encoder = nn.Sequential(
                # Block 1: Input -> 32 x 64 x 64
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 64 x 64

                # Block 2: 32 x 32 x 32
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 32 x 32

                # Block 3: 64 x 16 x 16
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 16 x 16

                # Block 4: 128 x 8 x 8
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 8 x 8

                # Block 5: 256 x 8 x 8 (No MaxPooling)
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)  # Switched to LeakyReLU
            )
            
            self.conv_to_latent_img = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.Conv2d(384, 384, kernel_size=2, bias=True)  # To reduce to (batchsize, n, 1, 1)
            )
            
            self.conv_to_latent_scat = nn.Sequential(
                nn.Conv2d(scatshape[-3], 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(0.2),  # Switched to LeakyReLU
                nn.Conv2d(384, 384, kernel_size=2, bias=True)  # To reduce to (batchsize, n, 1, 1)
            )
            
            self.FC_input = nn.Linear(768, 384, bias=True)  # Update input size
            self.bn1 = nn.BatchNorm1d(384)
            
            self.FC_hidden = nn.Linear(384, 256, bias=True)
            self.bn2 = nn.BatchNorm1d(256)
            
            self.FC_mean = nn.Linear(256, latent_dim, bias=True)
            self.FC_var = nn.Linear(256, latent_dim, bias=True)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, img, scat):
            cnn_features = self.cnn_encoder(img)
            img_h = self.conv_to_latent_img(cnn_features)
            scat_h = self.conv_to_latent_scat(scat)
            h = torch.cat((img_h, scat_h), dim=1)    
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.bn1(self.FC_input(h)))
            h = self.dropout(h)
            h = self.LeakyReLU(self.bn2(self.FC_hidden(h)))  # Replaced LayerNorm with BatchNorm1d
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar
        
        
    
class CNN_Decoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dim=256):
        super(CNN_Decoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim * 4 * 4, bias=True),
            nn.LeakyReLU(0.2),  # Switched to LeakyReLU
        )
        
        self.deconv_layers = nn.Sequential(
            # Upsampling 1: 4x4 -> 8x8
            nn.ConvTranspose2d(intermediate_dim, intermediate_dim // 2, kernel_size=4, stride=2, padding=1, bias=True),  # 8x8
            nn.BatchNorm2d(intermediate_dim // 2),
            nn.LeakyReLU(0.2),  # Switched to LeakyReLU
            
            nn.Conv2d(intermediate_dim // 2, intermediate_dim // 2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(intermediate_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(intermediate_dim // 2, intermediate_dim // 2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(intermediate_dim // 2),
            nn.LeakyReLU(0.2),

            # Upsampling 2: 8x8 -> 16x16
            nn.ConvTranspose2d(intermediate_dim // 2, 128, kernel_size=4, stride=2, padding=1, bias=True),  # 16x16 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),  # Switched to LeakyReLU
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Upsampling 3: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),  # 32x32 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),  # Switched to LeakyReLU
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Upsampling 4: 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True),  # 64x64 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),  # Switched to LeakyReLU
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # Upsampling 5: 64x64 -> 128x128
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=True),  # 128x128 
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc_layers(z)
        z = z.view(z.size(0), -1, 4, 4)  # Reshape to 4x4 feature map
        z = self.deconv_layers(z)
        return z



if False: # For [-1, 1] normalised images
    class MLP_Encoder(nn.Module):
        def __init__(self, input_channels, hidden_dim1=256, hidden_dim2=128, latent_dim=32):
            super(MLP_Encoder, self).__init__()

            # Adjust conv_to_latent layers to match channel dimensions correctly
            self.conv_to_latent = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim2),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(hidden_dim2, hidden_dim1, kernel_size=3, stride=2, padding=1),  # Updated to use hidden_dim2 as input channels
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=2)  # Final layer keeps the same channels
            )

            self.FC_input = nn.Linear(hidden_dim1, hidden_dim1)
            self.bn1 = nn.BatchNorm1d(hidden_dim1)
            
            self.FC_hidden = nn.Linear(hidden_dim1, hidden_dim2)
            self.bn2 = nn.BatchNorm1d(hidden_dim2)
            
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)
            self.layer_norm = nn.LayerNorm(hidden_dim2)

        def forward(self, x):
            h = self.conv_to_latent(x)
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.bn1(self.FC_input(h)))
            h = self.dropout(h)
            h = self.LeakyReLU(self.layer_norm(self.FC_hidden(h)))  # Using LayerNorm instead of BatchNorm
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar
        

    class CNN_Encoder(nn.Module):
        def __init__(self, hidden_dim1, hidden_dim2, latent_dim):
            super(CNN_Encoder, self).__init__()
            
            # Convolutional layers
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 64 x 64
                
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 32 x 32
                
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 16 x 16
                
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 8 x 8
                
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)  # Changed to LeakyReLU
            )
            
            # Additional convolutional layers to reduce dimensions
            self.conv_to_latent = nn.Sequential(
                nn.Conv2d(256, hidden_dim2, kernel_size=3, stride=2, padding=1),  # Output: (hidden_dim2, 4, 4)
                nn.BatchNorm2d(hidden_dim2),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(hidden_dim2, hidden_dim1, kernel_size=3, stride=2, padding=1),  # Output: (hidden_dim1, 2, 2)
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(hidden_dim1, latent_dim, kernel_size=2)  # Output: (latent_dim, 1, 1)
            )
            
            self.fc_layers = MLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim)

            
        def forward(self, x):
            x = self.conv_layers(x)
            mean, logvar = self.fc_layers(x) 
            return mean, logvar


    class Dual_Encoder(nn.Module):
        def __init__(self, scatshape, hidden_dim1, hidden_dim2, latent_dim):
            super(Dual_Encoder, self).__init__()
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 64 x 64
                
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 32 x 32
                
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 16 x 16
                
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256 x 8 x 8
            )
            self.conv_to_latent_img = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(384, 384, kernel_size=2)  # To reduce to (batchsize, n, 1, 1)
            )
            self.conv_to_latent_scat = nn.Sequential(
                nn.Conv2d(scatshape[-3], 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(384, 384, kernel_size=2)  # To reduce to (batchsize, n, 1, 1)
            )
            
            self.FC_input = nn.Linear(768, hidden_dim2)  # Update input size
            self.bn1 = nn.BatchNorm1d(hidden_dim2)
            
            self.FC_hidden = nn.Linear(hidden_dim2, hidden_dim2)
            self.bn2 = nn.BatchNorm1d(hidden_dim2)
            
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)
            self.layer_norm = nn.LayerNorm(hidden_dim2)
            
        def forward(self, img, scat):
            cnn_features = self.cnn_encoder(img)
            img_h = self.conv_to_latent_img(cnn_features)
            scat_h = self.conv_to_latent_scat(scat)
            h = torch.cat((img_h, scat_h), dim=1)    
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.bn1(self.FC_input(h)))
            h = self.dropout(h)
            h = self.LeakyReLU(self.layer_norm(self.FC_hidden(h)))  # Using LayerNorm instead of BatchNorm
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar


        
    ###################################################
    ############### DECODERS ##########################
    ################################################### 

    class CNN_Decoder(nn.Module):
        def __init__(self, latent_dim, intermediate_dim=256):
            super(CNN_Decoder, self).__init__()
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, intermediate_dim * 4 * 4),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
            )
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(intermediate_dim, intermediate_dim // 2, kernel_size=4, stride=2, padding=1),  # 8x8
                nn.BatchNorm2d(intermediate_dim // 2),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.ConvTranspose2d(intermediate_dim // 2, 128, kernel_size=4, stride=2, padding=1),  # 16x16 
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32 
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64 
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 128x128 
                #nn.Sigmoid()
                nn.Tanh()
            )

        def forward(self, z):
            z = self.fc_layers(z)
            z = z.view(z.size(0), -1, 4, 4)  # Reshape to 8x8 feature mapshape 
            z = self.deconv_layers(z)
            #z = 2 * z - 1  # Scale output to [-1, 1]
            return z
        
if False: #Batchnorm everywhere Tanh
    class MLP_Encoder(nn.Module):
        def __init__(self, input_channels, hidden_dim1=256, hidden_dim2=128, latent_dim=32):
            super(MLP_Encoder, self).__init__()

            # Adjust conv_to_latent layers to match channel dimensions correctly
            self.conv_to_latent = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim2),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(hidden_dim2, hidden_dim1, kernel_size=3, stride=2, padding=1),  # Updated to use hidden_dim2 as input channels
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=2)  # Final layer keeps the same channels
            )

            self.FC_input = nn.Linear(hidden_dim1, hidden_dim1)
            self.bn1 = nn.BatchNorm1d(hidden_dim1)
            
            self.FC_hidden = nn.Linear(hidden_dim1, hidden_dim2)
            self.bn2 = nn.BatchNorm1d(hidden_dim2)
            
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            h = self.conv_to_latent(x)
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.bn1(self.FC_input(h)))
            h = self.dropout(h)
            h = self.LeakyReLU(self.bn2(self.FC_hidden(h)))  # Using BatchNorm1d instead of LayerNorm
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar


    class CNN_Encoder(nn.Module):
        def __init__(self, latent_dim):
            super(CNN_Encoder, self).__init__()
            
            # Convolutional layers
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 64 x 64
                
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 32 x 32
                
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 16 x 16
                
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 8 x 8
                
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)  # Changed to LeakyReLU
            )
            
            # Additional convolutional layers to reduce dimensions
            self.conv_to_latent = nn.Sequential(
                nn.Conv2d(256, hidden_dim2, kernel_size=3, stride=2, padding=1),  # Output: (hidden_dim2, 4, 4)
                nn.BatchNorm2d(hidden_dim2),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(hidden_dim2, hidden_dim1, kernel_size=3, stride=2, padding=1),  # Output: (hidden_dim1, 2, 2)
                nn.BatchNorm2d(hidden_dim1),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(hidden_dim1, latent_dim, kernel_size=2)  # Output: (latent_dim, 1, 1)
            )
            
            self.fc_layers = MLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim)

            
        def forward(self, x):
            x = self.conv_layers(x)
            mean, logvar = self.fc_layers(x) 
            return mean, logvar


    class Dual_Encoder(nn.Module):
        def __init__(self, scatshape, latent_dim):
            super(Dual_Encoder, self).__init__()
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 64 x 64
                
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 32 x 32
                
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 16 x 16
                
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256 x 8 x 8
            )
            self.conv_to_latent_img = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(384, 384, kernel_size=2)  # To reduce to (batchsize, n, 1, 1)
            )
            self.conv_to_latent_scat = nn.Sequential(
                nn.Conv2d(scatshape[-3], 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Conv2d(384, 384, kernel_size=2)  # To reduce to (batchsize, n, 1, 1)
            )
            
            self.FC_input = nn.Linear(768, hidden_dim2)  # Update input size
            self.bn1 = nn.BatchNorm1d(hidden_dim2)
            
            self.FC_hidden = nn.Linear(hidden_dim2, hidden_dim2)
            self.bn2 = nn.BatchNorm1d(hidden_dim2)
            
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, img, scat):
            cnn_features = self.cnn_encoder(img)
            img_h = self.conv_to_latent_img(cnn_features)
            scat_h = self.conv_to_latent_scat(scat)
            h = torch.cat((img_h, scat_h), dim=1)    
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.bn1(self.FC_input(h)))
            h = self.dropout(h)
            h = self.LeakyReLU(self.bn2(self.FC_hidden(h)))  # Using BatchNorm1d instead of LayerNorm
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar


    ###################################################
    ############### DECODERS ##########################
    ################################################### 

    class CNN_Decoder(nn.Module):
        def __init__(self, latent_dim, intermediate_dim=256):
            super(CNN_Decoder, self).__init__()
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, intermediate_dim * 4 * 4),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
            )
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(intermediate_dim, intermediate_dim // 2, kernel_size=4, stride=2, padding=1),  # 8x8
                nn.BatchNorm2d(intermediate_dim // 2),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.ConvTranspose2d(intermediate_dim // 2, 128, kernel_size=4, stride=2, padding=1),  # 16x16 
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32 
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64 
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 128x128 
                #nn.Sigmoid()
                nn.Tanh()
            )

        def forward(self, z):
            z = self.fc_layers(z)
            z = z.view(z.size(0), -1, 4, 4)  # Reshape to 8x8 feature mapshape 
            z = self.deconv_layers(z)
            #z = 2 * z - 1  # Scale output to [-1, 1]
            return z


    
class VAE_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
    
    def kl_divergence(self, x, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / x.size(0)
    
class Dual_VAE_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Dual_VAE_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, img, scat):
        mean, logvar = self.encoder(img, scat)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
    
    def kl_divergence(self, x, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / x.size(0)

###############################################
############ CONDITIONAL CNN VAE ##############
###############################################

class CMLP_Encoder(nn.Module):
    def __init__(self, input_channels, condition_dim, hidden_dim1=256, hidden_dim2=128, latent_dim=32):
        super(CMLP_Encoder, self).__init__()

        self.conv_to_latent = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim2, hidden_dim1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=2)
        )

        self.FC_input = nn.Linear(hidden_dim1 + condition_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)

        self.FC_hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

        self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
        self.FC_var = nn.Linear(hidden_dim2, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim2)

    def forward(self, x, condition):
        h = self.conv_to_latent(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, condition], dim=1)  # Concatenate condition with latent features
        h = self.LeakyReLU(self.bn1(self.FC_input(h)))
        h = self.dropout(h)
        h = self.LeakyReLU(self.layer_norm(self.FC_hidden(h)))
        h = self.dropout(h)
        mean = self.FC_mean(h)
        logvar = self.FC_var(h)
        return mean, logvar

class CCNN_Encoder(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(CCNN_Encoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv_to_latent = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, latent_dim, kernel_size=2)
        )

        self.fc_layers = CMLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim, condition_dim=condition_dim)

    def forward(self, x, condition):
        x = self.conv_layers(x)
        mean, logvar = self.fc_layers(x, condition)
        return mean, logvar

class CDual_Encoder(nn.Module):
    def __init__(self, scatshape, hidden_dim1, hidden_dim2, latent_dim, condition_dim):
        super(CDual_Encoder, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=2)
        )
        self.conv_to_latent_scat = nn.Sequential(
            nn.Conv2d(scatshape[-3], 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=2)
        )

        self.FC_input = nn.Linear(768 + condition_dim, hidden_dim2)
        self.bn1 = nn.BatchNorm1d(hidden_dim2)

        self.FC_hidden = nn.Linear(hidden_dim2, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

        self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
        self.FC_var = nn.Linear(hidden_dim2, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim2)
        #self.condition_fc = nn.Linear(128, 81)


    def forward(self, img, scat, condition):
        cnn_features = self.cnn_encoder(img)
        img_h = self.conv_to_latent_img(cnn_features)
        scat_h = self.conv_to_latent_scat(scat)
        h = torch.cat((img_h, scat_h), dim=1)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, condition], dim=1)
        h = self.LeakyReLU(self.bn1(self.FC_input(h)))
        h = self.dropout(h)
        h = self.LeakyReLU(self.layer_norm(self.FC_hidden(h)))
        h = self.dropout(h)
        mean = self.FC_mean(h)
        logvar = self.FC_var(h)
        return mean, logvar
    
class CCNN_Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, intermediate_dim=None):
        if intermediate_dim is None:
            intermediate_dim = latent_dim + condition_dim
        super(CCNN_Decoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, intermediate_dim * 4 * 4),
            nn.ReLU(),
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(intermediate_dim, intermediate_dim // 2, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(intermediate_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(intermediate_dim // 2, 128, kernel_size=4, stride=2, padding=1),  # 16x16 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 128x128 
            nn.Sigmoid()
        )

    def forward(self, z, condition):
        z = torch.cat([z, condition], dim=1)
        z = self.fc_layers(z)
        z = z.view(z.size(0), -1, 4, 4)  # Reshape to the feature map shape (batch_size, channels, height, width)
        z = self.deconv_layers(z)
        return z


class CVAE_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, condition):
        mean, logvar = self.encoder(x, condition)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z, condition=condition)
        return x_hat, mean, logvar

    def kl_divergence(self, x, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / x.size(0)

class CDual_VAE_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(CDual_VAE_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, img, scat, condition):
        mean, logvar = self.encoder(img, scat, condition)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z, condition)
        return x_hat, mean, logvar
    
    def kl_divergence(self, x, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / x.size(0)


##############################################


class Alex_VAE_Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Alex_VAE_Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Output: 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_to_latent = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # Output: 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, latent_dim * 2, kernel_size=4)  # Output: latent_dim * 2 x 1 x 1
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv_to_latent(x)
        x = x.view(x.size(0), -1)
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

class Alex_VAE_Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Alex_VAE_Decoder, self).__init__()
        self.latent_to_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4),  # Output: 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 1 x 128 x 128
            nn.Sigmoid()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        z = self.latent_to_conv(z)
        z = self.deconv_layers(z)
        return z

class Alex_VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(Alex_VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
    
    def kl_divergence(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld
    
########################################################
    

class EfficientNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(EfficientNetEncoder, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)
        self.efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc = nn.Linear(1280, latent_dim * 2)

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

class EfficientNetDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape=(1, 128, 128)):
        super(EfficientNetDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 1280)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, output_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Ensure output is in range [0, 1]
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 1280, 1, 1)
        z = nn.functional.interpolate(z, scale_factor=8, mode='nearest')
        x_hat = self.decoder(z)
        return x_hat

class EfficientNetVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(EfficientNetVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
    
def get_model(name, scatshape, hidden_dim1=None, hidden_dim2=None, latent_dim=None, num_classes=4):
    if 'MLP' in name and 'C' not in name:
        encoder = MLP_Encoder(input_channels=scatshape[-3], hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim)
        decoder = CNN_Decoder(latent_dim=latent_dim, intermediate_dim=128)
        model = VAE_Model(encoder=encoder, decoder=decoder)
    elif 'MLP' in name: #    elif name in ["CSTMLP", "ClavgSTMLP", "CldiffSTMLP"]:
        encoder = CMLP_Encoder(input_channels=scatshape[-3], hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, condition_dim=num_classes)
        decoder = CCNN_Decoder(latent_dim=latent_dim, condition_dim=num_classes) 
        model = CVAE_Model(encoder=encoder, decoder=decoder)
    elif name == "CNN":
        encoder = CNN_Encoder(latent_dim=latent_dim)
        decoder = CNN_Decoder(latent_dim=latent_dim, intermediate_dim=128)
        model = VAE_Model(encoder=encoder, decoder=decoder)
    elif name == 'DualCNN' or name == 'Dual':
        encoder = Dual_Encoder(scatshape=scatshape, latent_dim=latent_dim)
        decoder = CNN_Decoder(latent_dim=latent_dim, intermediate_dim=128)
        model = Dual_VAE_Model(encoder=encoder, decoder=decoder)
    elif name == 'Alex':
        encoder = Alex_VAE_Encoder(latent_dim=latent_dim)
        decoder = Alex_VAE_Decoder(latent_dim=latent_dim)
        model = Alex_VAE(encoder=encoder, decoder=decoder)
    elif name == "CAlex":
        encoder = ConditionalAlexVAEEncoder(latent_dim=latent_dim, num_classes=num_classes)
        decoder = ConditionalAlexVAEDecoder(latent_dim=latent_dim, num_classes=num_classes)
        model = ConditionalAlexVAE(encoder=encoder, decoder=decoder)
    elif name == "CCNN":
        encoder = CCNN_Encoder(latent_dim=latent_dim, condition_dim=num_classes)
        decoder = CCNN_Decoder(latent_dim=latent_dim, condition_dim=num_classes)
        model = CVAE_Model(encoder=encoder, decoder=decoder)
    elif name == "CDual":
        encoder = CDual_Encoder(scatshape=scatshape, hidden_dim1=512, hidden_dim2=256, latent_dim=latent_dim, condition_dim=num_classes)
        decoder = CCNN_Decoder(latent_dim=latent_dim, condition_dim=num_classes)
        model = CDual_VAE_Model(encoder=encoder, decoder=decoder)
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    return model

    
###############################################
################# CLASSIFIERS #################
###############################################

    
class CNN_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Classifier, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # Assuming grayscale images as input
            nn.BatchNorm2d(32),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(64 * 8 * 8, 64 * 8 * 8),  # Adjust the input size according to the output feature map size
            nn.Tanh(),
            nn.Linear(64 * 8 * 8, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64, 64),  # Reduce the number of parameters by decreasing the number of units
            nn.BatchNorm1d(64),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Add dropout with 50% probability
            nn.Linear(64, num_classes)  # Ten outputs for ten classes
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        #attn_weights = self.attention(x)
        #x = x * attn_weights
        x = self.classifier(x)
        return x


############################# UTILS




class NormalisedWeightedMSELoss(nn.Module):
    """A custom loss function that calculates the MSE loss with the sum of the relevant pixels weighted by a factor"""
    def __init__(self, threshold=0.2, weight=10.0):
        super(NormalisedWeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, input, target, return_map=False):
        # Create mask and initial weight map
        mask = (target > self.threshold).float()
        weight_map = torch.ones_like(target)
        num_masked_elements = mask.sum()
        num_unmasked_elements = mask.numel() - num_masked_elements

        # Normalize the weight_map to ensure the total weight of the region above the threshold is 10 times the total weight of the rest
        if num_unmasked_elements > 0:
            total_weight_masked, total_weight_unmasked = self.weight, 1
            weight_map[mask == 1] = total_weight_masked / num_masked_elements
            weight_map[mask == 0] = total_weight_unmasked / num_unmasked_elements

        # Normalize so that the sum of the weight map is 1
        weight_map = weight_map / weight_map.sum()

        # Calculate loss
        loss = nn.functional.mse_loss(input, target, reduction='none') * weight_map
        
        if return_map:
            return loss, weight_map
        else:
            return loss.sum()


class RadialWeightedMSELoss(nn.Module):
    """A custom loss function that calculates the MSE loss with the sum of the relevant pixels weighted by a factor and adds a radial weight"""
    def __init__(self, threshold=0.2, intensity_weight=1.0, radial_weight=1.0):
        super(RadialWeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.intensity_weight = intensity_weight
        self.radial_weight = radial_weight

    def forward(self, input, target, return_map=False):
        # Ensure input and target are 4D tensors
        if input.dim() == 3:
            input = input.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Create mask and map
        mask = (target > self.threshold).float()
        intensity_map = (target * mask) ** self.intensity_weight

        # Calculate radial distance map
        _, _, height, width = target.shape
        y_center, x_center = height // 2, width // 2
        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        y_grid = y_grid.to(target.device).float()
        x_grid = x_grid.to(target.device).float()
        distance_map = torch.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)
        max_distance = torch.max(distance_map)
        radial_weight_map = (1.0 - (distance_map / max_distance)) ** self.radial_weight

        weight_map = intensity_map * radial_weight_map
        loss_map = nn.functional.mse_loss(input, target, reduction='none') * weight_map
        zero_pixels_loss_map = nn.functional.mse_loss(input * (1 - mask), target * (1 - mask), reduction='none')

        total_loss_map = loss_map + zero_pixels_loss_map
        
        if return_map:
            return total_loss_map, weight_map
        else:
            return total_loss_map.sum()



class CustomIntensityWeightedMSELoss(nn.Module):
    """Weighting the intensity distribution too"""
    def __init__(self, intensity_threshold=0.2, intensity_weight=2.0, log_weight=1.0):
        super(CustomIntensityWeightedMSELoss, self).__init__()
        self.intensity_threshold = intensity_threshold
        self.intensity_weight = intensity_weight
        self.log_weight = log_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, generated, target, return_map=False):
        # Ensure that generated and target have the correct dimensions
        if generated.dim() == 3:
            generated = generated.unsqueeze(1)  # Add a channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add a channel dimension

        # Compute base MSE loss
        mse_loss = self.mse_loss(generated, target)

        # Apply intensity weighting
        intensity_mask = (target > self.intensity_threshold).float()
        weighted_mse_loss = mse_loss * (1 + intensity_mask * (self.intensity_weight - 1))

        # Apply logarithmic weighting
        log_diff = torch.abs(torch.log1p(generated) - torch.log1p(target))
        log_loss = self.log_weight * log_diff

        # Combine losses
        total_loss = weighted_mse_loss + log_loss

        if return_map:
            # Ensure that the loss map has the correct dimensions
            return total_loss, intensity_mask
        else:
            return total_loss.mean()



class WeightedMSELoss(nn.Module):
    """Weighting pixels above threshold more heavily"""
    def __init__(self, threshold=0.2, weight=1.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.intensity_weight = weight

    def forward(self, input, target, return_map=False):
        # Ensure that the input and target have the correct dimensions
        if input.dim() == 3:
            input = input.unsqueeze(1)  # Add a channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add a channel dimension

        mask = (target > self.threshold).float()
        intensity_map = (target * mask) ** self.intensity_weight
        
        # Compute the MSE loss for the masked regions
        mse_loss = nn.functional.mse_loss(input * mask * intensity_map, target * mask * intensity_map, reduction='none')
        zero_pixels_loss = nn.functional.mse_loss(input * (1 - mask), target * (1 - mask), reduction='none')

        # Combine the losses
        total_loss = mse_loss + zero_pixels_loss

        if return_map:
            # Ensure the loss map has the correct dimensions
            return total_loss, intensity_map
        else:
            return total_loss.sum()


class MaxIntensityMSELoss(nn.Module):
    def __init__(self, intensity_weight=1.0, sum_weight=1.0):
        super(MaxIntensityMSELoss, self).__init__()
        self.intensity_weight = intensity_weight
        self.sum_weight = sum_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input, target, return_map=False):
        # Calculate the standard MSE loss element-wise
        loss_map = self.mse_loss(input, target)
        weight_map = torch.ones_like(loss_map)

        # Calculate the difference of the maximum intensity pixels
        max_intensity_input = input.view(input.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_target = target.view(target.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_diff = torch.abs(max_intensity_input - max_intensity_target)

        # Calculate the difference of the summed pixel intensities
        sum_intensity_input = input.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_target = target.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_diff = torch.abs(sum_intensity_input - sum_intensity_target)

        # Add these terms to the loss
        max_intensity_penalty = self.intensity_weight * max_intensity_diff.mean()
        sum_intensity_penalty = self.sum_weight * sum_intensity_diff.mean()

        # Scale down the additional penalties
        combined_loss = loss_map + max_intensity_penalty + sum_intensity_penalty

        if return_map:
            return combined_loss, weight_map
        else:
            return combined_loss.sum()



class CustomMSELoss(nn.Module):
    def __init__(self, intensity_weight=0.001, sum_weight=0.001):
        super(CustomMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.intensity_weight = intensity_weight
        self.sum_weight = sum_weight
        self.epsilon = 1e-8  # Small constant to prevent division by zero

    def forward(self, input, target, return_map=False):
        # Calculate the standard MSE loss element-wise
        loss_map = self.mse_loss(input, target)
        weight_map = torch.ones_like(loss_map)
        weighted_loss_map = loss_map * weight_map
        
        # Calculate the root mean squared difference of the maximum intensity pixels
        max_intensity_input = input.view(input.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_target = target.view(target.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_diff = torch.sqrt(torch.abs(max_intensity_input - max_intensity_target) + self.epsilon).view(-1, 1, 1, 1)

        # Calculate the root mean squared difference of the summed pixel intensities
        sum_intensity_input = input.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_target = target.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_diff = torch.sqrt(torch.abs(sum_intensity_input - sum_intensity_target) + self.epsilon).view(-1, 1, 1, 1)

        # Normalize the penalties relative to the input size
        num_elements = torch.numel(input[0]) + self.epsilon  # Prevent division by zero
        max_intensity_diff = max_intensity_diff / num_elements
        sum_intensity_diff = sum_intensity_diff / num_elements

        # Expand the differences to match the loss map's shape
        max_intensity_diff = max_intensity_diff.expand_as(loss_map)
        sum_intensity_diff = sum_intensity_diff.expand_as(loss_map)

        # Combine the losses
        weighted_loss_map = weighted_loss_map + \
                            self.intensity_weight * max_intensity_diff + \
                            self.sum_weight * sum_intensity_diff

        if return_map:
            return weighted_loss_map, weight_map
        else:
            return weighted_loss_map.sum()

        
class StandardMSELoss(nn.Module):
    def __init__(self):
        super(StandardMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input, target, return_map=False):
        if return_map:
            # Compute the loss per pixel
            loss_map = self.mse_loss(input, target)
            weight_map = torch.ones_like(loss_map)
            weighted_loss_map = loss_map * weight_map
            return weighted_loss_map, weight_map
        else:
            # Return the sum of the loss across all pixels
            return self.mse_loss(input, target).sum()



class CombinedMSELoss(nn.Module):
    def __init__(self, intensity_weight=0.001, sum_weight=0.001, threshold=0.2, high_intensity_weight=1.0):
        super(CombinedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.intensity_weight = intensity_weight
        self.sum_weight = sum_weight
        self.threshold = threshold
        self.high_intensity_weight = high_intensity_weight

    def forward(self, input, target, return_map=False):
        # Ensure that the input and target have the correct dimensions
        if input.dim() == 3:
            input = input.unsqueeze(1)  # Add a channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add a channel dimension

        # Create mask and intensity map for high-intensity weighting
        mask = (target > self.threshold).float()
        intensity_map = (target * mask) ** self.high_intensity_weight

        # Compute the MSE loss for the masked and unmasked regions
        mse_loss = nn.functional.mse_loss(input * mask * intensity_map, target * mask * intensity_map, reduction='none')
        zero_pixels_loss = nn.functional.mse_loss(input * (1 - mask), target * (1 - mask), reduction='none')

        # Combine the pixel-wise losses
        total_loss = mse_loss + zero_pixels_loss

        # Calculate the non-spatial terms
        # Root mean squared difference of the maximum intensity pixels
        max_intensity_input = input.view(input.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_target = target.view(target.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_diff = torch.sqrt((max_intensity_input - max_intensity_target) ** 2).view(-1, 1, 1, 1)

        # Root mean squared difference of the summed pixel intensities
        sum_intensity_input = input.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_target = target.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_diff = torch.sqrt((sum_intensity_input - sum_intensity_target) ** 2).view(-1, 1, 1, 1)

        # Normalize the penalties relative to the input size
        max_intensity_diff = max_intensity_diff / torch.numel(input[0])
        sum_intensity_diff = sum_intensity_diff / torch.numel(input[0])

        # Expand the differences to match the loss map's shape
        max_intensity_diff = max_intensity_diff.expand_as(total_loss)
        sum_intensity_diff = sum_intensity_diff.expand_as(total_loss)

        # Combine the pixel-wise loss with the non-spatial terms
        total_loss = total_loss + \
                     self.intensity_weight * max_intensity_diff + \
                     self.sum_weight * sum_intensity_diff

        if return_map:
            return total_loss, intensity_map
        else:
            return total_loss.sum()

    
    
#################################################
########## OLD STUFF ############################
#################################################


class Small_MLP_Encoder(nn.Module):
    def __init__(self, input_dim=128**2, hidden_dim=128, latent_dim=32):
        super(Small_MLP_Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.FC_hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.LeakyReLU(self.bn1(self.FC_input(x)))
        h = self.dropout(h)
        h = self.LeakyReLU(self.layer_norm(self.FC_hidden1(h)))  # Using LayerNorm instead of BatchNorm
        h = self.dropout(h)
        mean = self.FC_mean(h)
        logvar = self.FC_var(h)
        return mean, logvar
    
class Small_MLP_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Small_MLP_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
    
###############################################


    
####################################################

    
    
###############################################
class ConditionalAlexVAEEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ConditionalAlexVAEEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_to_latent = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, latent_dim * 2, kernel_size=4)
        )
        self.class_embedding = nn.Linear(num_classes, 1024)

    def forward(self, x, labels):
        x = self.conv_layers(x)
        x = self.conv_to_latent(x)
        x = x.view(x.size(0), -1)
        
        class_embeds = self.class_embedding(labels.float())
        print(f"Encoder: x shape: {x.shape}, class_embeds shape: {class_embeds.shape}")
        
        x = torch.cat([x, class_embeds], dim=1)
        
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

class ConditionalAlexVAEDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ConditionalAlexVAEDecoder, self).__init__()
        self.latent_to_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 1024, kernel_size=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.class_embedding = nn.Linear(num_classes, latent_dim)

    def forward(self, z, labels):
        class_embeds = self.class_embedding(labels.float())
        
        z = torch.cat([z, class_embeds], dim=1)
        print(f"Decoder: z shape after concat: {z.shape}")
        
        z = z.view(z.size(0), -1, 1, 1)
        z = self.latent_to_conv(z)
        print(f"Decoder: z shape after latent_to_conv: {z.shape}")
        
        z = self.deconv_layers(z)
        return z

class ConditionalAlexVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConditionalAlexVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, labels):
        mean, logvar = self.encoder(x, labels)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z, labels)
        return x_hat, mean, logvar

    def kl_divergence(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld

#############################################
###############################################


class CdualAlexVAEEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes, scatter_dim):
        super(CdualAlexVAEEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Output: 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_to_latent = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # Output: 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, latent_dim * 2, kernel_size=4)  # Output: latent_dim * 2 x 1 x 1
        )
        print("Num classes: ", num_classes)
        self.class_embedding = nn.Embedding(num_classes, 128)
        self.scatter_fc = nn.Linear(scatter_dim, 128)

    def forward(self, x, labels, scatter_coeffs):
        x = self.conv_layers(x)
        x = self.conv_to_latent(x)
        x = x.view(x.size(0), -1)
        
        class_embeds = self.class_embedding(labels)
        scatter_embeds = self.scatter_fc(scatter_coeffs)
        
        x = torch.cat([x, class_embeds, scatter_embeds], dim=1)
        
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

class CdualAlexVAEDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes, scatter_dim):
        super(CdualAlexVAEDecoder, self).__init__()
        self.latent_to_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes + scatter_dim, 1024, kernel_size=4),  # Output: 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 1 x 128 x 128
            nn.Sigmoid()
        )
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        self.scatter_fc = nn.Linear(scatter_dim, latent_dim)

    def forward(self, z, labels, scatter_coeffs):
        class_embeds = self.class_embedding(labels)
        scatter_embeds = self.scatter_fc(scatter_coeffs)
        
        z = torch.cat([z, class_embeds, scatter_embeds], dim=1)
        z = z.view(z.size(0), -1, 1, 1)
        z = self.latent_to_conv(z)
        z = self.deconv_layers(z)
        return z

class CdualAlexVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConditionalAlexVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, labels, scatter_coeffs):
        mean, logvar = self.encoder(x, labels, scatter_coeffs)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z, labels, scatter_coeffs)
        return x_hat, mean, logvar
    
    def kl_divergence(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld
