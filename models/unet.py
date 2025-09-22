
# import torch
# import torch.nn as nn


# def get_time_embedding(time_steps, temb_dim):
#     r"""
#     Convert time steps tensor into an embedding using the
#     sinusoidal time embedding formula
#     :param time_steps: 1D tensor of length batch size
#     :param temb_dim: Dimension of the embedding
#     :return: BxD embedding representation of B time steps
#     """
#     assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
#     # Convert scalar to 1D tensor
#     if time_steps.dim() == 0:
#         time_steps = time_steps.unsqueeze(0)
#     # factor = 10000^(2i/d_model)
#     factor = 10000 ** ((torch.arange(
#         start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
#     )
    
#     # pos / factor
#     # timesteps B -> B, 1 -> B, temb_dim
#     t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
#     t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
#     return t_emb


# class DownBlock(nn.Module):
#     r"""
#     Down conv block with attention.
#     Sequence of following block
#     1. Resnet block with time embedding
#     2. Attention block
#     3. Downsample
#     """
    
#     def __init__(self, in_channels, out_channels, t_emb_dim,
#                  down_sample, num_heads, attn, norm_channels):
#         super().__init__()
        
#         self.down_sample = down_sample
#         self.attn = attn
        
        
#         self.t_emb_dim = t_emb_dim
#         self.resnet_conv_first = nn.Sequential(
#             nn.GroupNorm(norm_channels, in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, out_channels,
#                       kernel_size=3, stride=1, padding=1),
#         )

#         if self.t_emb_dim is not None:
#             self.t_emb_layers = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(self.t_emb_dim, out_channels)
#             )

#         self.resnet_conv_second = nn.Sequential(
#             nn.GroupNorm(norm_channels, out_channels),
#             nn.SiLU(),
#             nn.Conv2d(out_channels, out_channels,
#                       kernel_size=3, stride=1, padding=1),
#         )
        
        
#         if self.attn:
#             self.attention_norms = nn.GroupNorm(norm_channels, out_channels)
#             self.attentions = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                

#         self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

            
#         self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
#                                           4, 2, 1) if self.down_sample else nn.Identity()
    
#     def forward(self, x, t_emb=None):
#         out = x
        
#         # Resnet block of Unet
#         resnet_input = out
#         out = self.resnet_conv_first(out)
#         if t_emb is not None:
#             out = out + self.t_emb_layers(t_emb)[:, :, None, None]
#         out = self.resnet_conv_second(out)
#         out = out + self.residual_input_conv(resnet_input)

#         if self.attn:
#                 # Attention block of Unet

#             batch_size, channels, h, w = out.shape
#             in_attn = out.reshape(batch_size, channels, h * w)
#             in_attn = self.attention_norms(in_attn)
#             in_attn = in_attn.transpose(1, 2)
#             out_attn, _ = self.attentions(in_attn, in_attn, in_attn)
#             out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#             out = out + out_attn
                
#         # Downsample
#         out = self.down_sample_conv(out)
#         return out


# class UpBlockUnet(nn.Module):
#     r"""
#     Up conv block with attention.
#     Sequence of following blocks
#     1. Upsample
#     1. Concatenate Down block output
#     2. Resnet block with time embedding
#     3. Attention Block
#     """
    
#     def __init__(self, in_channels, out_channels, t_emb_dim, up_sample,
#                  num_heads, attn, norm_channels):
#         super().__init__()
#         self.up_sample = up_sample
#         self.attn = attn
#         self.t_emb_dim = t_emb_dim
#         self.resnet_conv_first = nn.Sequential(
#             nn.GroupNorm(norm_channels, in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#         )


#         if self.t_emb_dim is not None:
#             self.t_emb_layers = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(t_emb_dim, out_channels)
#             )

#         self.resnet_conv_second = nn.Sequential(
#             nn.GroupNorm(norm_channels, out_channels),
#             nn.SiLU(),
#             nn.Conv2d(out_channels, out_channels,
#                       kernel_size=3, stride=1, padding=1),
#         )
        
#         if self.attn:
#             self.attention_norms = nn.GroupNorm(norm_channels, out_channels)
#             self.attentions = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                

#         self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#         self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels ,
#                                                  4, 2, 1) \
#             if self.up_sample else nn.Identity()
    
#     def forward(self, x, out_down=None, t_emb=None):
        
#         if out_down is not None:
#             assert x.shape == out_down.shape
#             x = torch.cat([x, out_down], dim=1)
#         x = self.up_sample_conv(x)
#         out = x
        
#            # Resnet
#         resnet_input = out
#         out = self.resnet_conv_first(out)
#         if self.t_emb_dim is not None:
#             out = out + self.t_emb_layers(t_emb)[:, :, None, None]
#         out = self.resnet_conv_second(out)
#         out = out + self.residual_input_conv(resnet_input)
#         if self.attn:
#             # Self Attention
#             batch_size, channels, h, w = out.shape
#             in_attn = out.reshape(batch_size, channels, h * w)
#             in_attn = self.attention_norms(in_attn)
#             in_attn = in_attn.transpose(1, 2)
#             out_attn, _ = self.attentions(in_attn, in_attn, in_attn)
#             out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#             out = out + out_attn
            
#         return out
    

# class Encoder(nn.Module):
#     def __init__(self, im_channels=3):
#         super().__init__()
#         self.initial_conv = nn.Sequential(
#             nn.Conv2d(im_channels, 64, 3, 1, 1),
#             nn.GroupNorm(2, 64),
#             nn.SiLU()
#         )
        
#         self.down1 =DownBlock(64, 128, t_emb_dim=None,down_sample= True,num_heads=16,attn= False, norm_channels=2)
        
#         #self.down2 = DownBlock(128, 128,t_emb_dim=None,down_sample= False,num_heads=16,attn= False, norm_channels=8)
        
#         self.down3 = DownBlock(128, 256,t_emb_dim=None,down_sample= True,num_heads=16,attn= False, norm_channels=8)

#         #self.down4 = DownBlock(256, 256, t_emb_dim=None,down_sample= False,num_heads=16,attn= False, norm_channels=8)

#         self.down5 = DownBlock(256, 256, t_emb_dim=None,down_sample= True,num_heads=16,attn= False, norm_channels=8)

#         self.down6 = DownBlock(256, 384, t_emb_dim=None,down_sample=False ,num_heads=16,attn= False, norm_channels=8)

#         self.down7 = DownBlock(384, 512, t_emb_dim=None,down_sample=True ,num_heads=16,attn= False, norm_channels=8)
        
#     def forward(self, x):
#         skip_connections = []
#         x = self.initial_conv(x)
#         x = self.down1(x)
#         #x = self.down2(x)
#         x = self.down3(x)
#         #x = self.down4(x)
#         x = self.down5(x); skip_connections.append(x)
#         x = self.down6(x); skip_connections.append(x)
#         x = self.down7(x); skip_connections.append(x)
#         return skip_connections


# class Unet(nn.Module):
#     r"""
#     Unet model comprising
#     Down blocks, Midblocks and Uplocks
#     """
    
#     def __init__(self, im_channels):
#         super().__init__()
        
#         self.conv_in = nn.Conv2d(8, 256, kernel_size=3, padding=1)
        
#         # Initial projection from sinusoidal time embedding
#         self.t_proj = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.SiLU(),
#             nn.Linear(128, 128)
#         )
        
#         #self.down1=DownBlock(256, 256, 128,down_sample= False,num_heads=16,attn= False, norm_channels=8)

#         self.down2=DownBlock(256, 384, 128,down_sample= False,num_heads=16,attn= False, norm_channels=8)

#         self.down3=DownBlock(384, 512, 128,down_sample= True,num_heads=16,attn= True, norm_channels=8)
        
#         #self.up1=UpBlockUnet(512, 512, 128, up_sample= False,num_heads=16, norm_channels=8)
        
#         self.up2=UpBlockUnet(512, 384, 128, up_sample= True,num_heads=16,attn= True, norm_channels=8)
        
#         self.up3=UpBlockUnet(768, 256, 128, up_sample= False,num_heads=16,attn= False, norm_channels=8)
        
#         #self.up4=UpBlockUnet(512, 256, 128, up_sample= False,num_heads=16,attn= False, norm_channels=8)
        
                           
        
#         self.norm_out = nn.GroupNorm(8, 512)
#         self.conv_out = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        

#     def forward(self, x, t, encoder_skips=None):
        
#         out = self.conv_in(x)
        
#         enc_sk1, enc_sk2, enc_sk3 = None, None, None
#         if encoder_skips is not None:
#             if len(encoder_skips) > 0:
#                 enc_sk1 = encoder_skips[0]
#             if len(encoder_skips) > 1:
#                 enc_sk2 = encoder_skips[1]
#             if len(encoder_skips) > 2:
#                 enc_sk3 = encoder_skips[2]    
#         # t_emb -> B x t_emb_dim
#         t_emb = get_time_embedding(torch.as_tensor(t).long(), 128)
#         t_emb = self.t_proj(t_emb)
        
#         down_outs = []
        
#         down_outs.append(out)
        
#         assert out.shape == enc_sk1.shape
#         out=out+enc_sk1
# #1st
#         # out = self.down1(out, t_emb)
#         # down_outs.append(out)
#         # assert out.shape == enc_sk2.shape
#         # out=out+enc_sk2
        
        
#         out = self.down2(out, t_emb)
#         down_outs.append(out)
#         assert out.shape == enc_sk2.shape
#         out=out+enc_sk2

#         out = self.down3(out, t_emb)
        
#         #up
#         #out = self.up1(out, None, t_emb)
        
#         assert out.shape == enc_sk3.shape
#         out=out+enc_sk3
#         out= self.up2(out, None, t_emb)

        
#         out = self.up3(out, down_outs.pop(), t_emb)
        
#         #out = self.up4(out, down_outs.pop(), t_emb)

#         out = torch.cat([out, down_outs.pop()], dim=1)

#         out = self.norm_out(out)
#         out = nn.SiLU()(out)
#         out = self.conv_out(out)
#         #print("out shape from unet", out.shape)
#         # out B x C x H x W
#         return out

import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # Convert scalar to 1D tensor
    if time_steps.dim() == 0:
        time_steps = time_steps.unsqueeze(0)
    
    # Use float timesteps instead of long for better precision
    time_steps = time_steps.float()
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization with time conditioning"""
    def __init__(self, num_groups, num_channels, t_emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.time_proj = nn.Linear(t_emb_dim, num_channels * 2)  # scale and shift
        
    def forward(self, x, t_emb):
        x = self.norm(x)
        if t_emb is not None:
            params = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # B, 2*C, 1, 1
            scale, shift = params.chunk(2, dim=1)
            x = x * (1 + scale) + shift
        return x


class DownBlock(nn.Module):
    r"""
    Down conv block with attention and improved time conditioning.
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, attn, norm_channels):
        super().__init__()
        
        self.down_sample = down_sample
        self.attn = attn
        self.t_emb_dim = t_emb_dim
        
        # Use AdaptiveGroupNorm for better time conditioning
        if self.t_emb_dim is not None:
            self.norm1 = AdaptiveGroupNorm(norm_channels, in_channels, t_emb_dim)
            self.norm2 = AdaptiveGroupNorm(norm_channels, out_channels, t_emb_dim)
        else:
            self.norm1 = nn.GroupNorm(norm_channels, in_channels)
            self.norm2 = nn.GroupNorm(norm_channels, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.activation = nn.SiLU()
        
        # Enhanced time embedding layers
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.t_emb_dim, out_channels),
                nn.SiLU(),
                nn.Linear(out_channels, out_channels)
            )
        
        if self.attn:
            self.attention_norms = nn.GroupNorm(norm_channels, out_channels)
            self.attentions = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb=None):
        out = x
        
        # Resnet block with improved time conditioning
        resnet_input = out
        
        # First conv block
        if self.t_emb_dim is not None and t_emb is not None:
            out = self.norm1(out, t_emb)
        else:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv1(out)
        
        # Apply time embedding after first conv
        if t_emb is not None and self.t_emb_dim is not None:
            time_emb = self.t_emb_layers(t_emb)[:, :, None, None]
            out = out + time_emb
        
        # Second conv block
        if self.t_emb_dim is not None and t_emb is not None:
            out = self.norm2(out, t_emb)
        else:
            out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        # Residual connection
        out = out + self.residual_input_conv(resnet_input)

        # Attention block
        if self.attn:
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms(in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions(in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
                
        # Downsample
        out = self.down_sample_conv(out)
        return out


class UpBlockUnet(nn.Module):
    r"""
    Up conv block with attention and improved time conditioning.
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample,
                 num_heads, attn, norm_channels):
        super().__init__()
        self.up_sample = up_sample
        self.attn = attn
        self.t_emb_dim = t_emb_dim
        
        # Use AdaptiveGroupNorm for better time conditioning
        if self.t_emb_dim is not None:
            self.norm1 = AdaptiveGroupNorm(norm_channels, in_channels, t_emb_dim)
            self.norm2 = AdaptiveGroupNorm(norm_channels, out_channels, t_emb_dim)
        else:
            self.norm1 = nn.GroupNorm(norm_channels, in_channels)
            self.norm2 = nn.GroupNorm(norm_channels, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.activation = nn.SiLU()
        
        # Enhanced time embedding layers
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels),
                nn.SiLU(),
                nn.Linear(out_channels, out_channels)
            )
        
        if self.attn:
            self.attention_norms = nn.GroupNorm(norm_channels, out_channels)
            self.attentions = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1) if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down=None, t_emb=None):
        
        if out_down is not None:
            assert x.shape == out_down.shape
            x = torch.cat([x, out_down], dim=1)
        x = self.up_sample_conv(x)
        out = x
        
        # Resnet block with improved time conditioning
        resnet_input = out
        
        # First conv block
        if self.t_emb_dim is not None and t_emb is not None:
            out = self.norm1(out, t_emb)
        else:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv1(out)
        
        # Apply time embedding after first conv
        if self.t_emb_dim is not None and t_emb is not None:
            time_emb = self.t_emb_layers(t_emb)[:, :, None, None]
            out = out + time_emb
        
        # Second conv block
        if self.t_emb_dim is not None and t_emb is not None:
            out = self.norm2(out, t_emb)
        else:
            out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        # Residual connection
        out = out + self.residual_input_conv(resnet_input)
        
        # Attention block
        if self.attn:
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms(in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions(in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            
        return out


class Encoder(nn.Module):
    def __init__(self, im_channels=3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(im_channels, 64, 3, 1, 1),
            nn.GroupNorm(2, 64),
            nn.SiLU()
        )
        
        # Note: These don't use time embeddings, so t_emb_dim=None
        self.down1 = DownBlock(64, 128, t_emb_dim=None, down_sample=True, num_heads=16, attn=False, norm_channels=2)
        self.down3 = DownBlock(128, 256, t_emb_dim=None, down_sample=True, num_heads=16, attn=False, norm_channels=8)
        self.down4 = DownBlock(256, 256, t_emb_dim=None, down_sample=True, num_heads=16, attn=False, norm_channels=8)
        self.down5 = DownBlock(256, 256, t_emb_dim=None, down_sample=False, num_heads=16, attn=False, norm_channels=8)
        self.down6 = DownBlock(256, 384, t_emb_dim=None, down_sample=False, num_heads=16, attn=False, norm_channels=8)
        self.down7 = DownBlock(384, 512, t_emb_dim=None, down_sample=True, num_heads=16, attn=False, norm_channels=8)
        
    def forward(self, x):
        skip_connections = []
        x = self.initial_conv(x)
        x = self.down1(x)
        x = self.down3(x)
        x = self.down4(x); skip_connections.append(x)
        x = self.down5(x); skip_connections.append(x)
        x = self.down6(x); skip_connections.append(x)
        x = self.down7(x); skip_connections.append(x)
        return skip_connections


class Unet(nn.Module):
    r"""
    Unet model with improved time embedding and conditioning
    """
    
    def __init__(self, im_channels):
        super().__init__()
        
        self.conv_in = nn.Conv2d(264, 256, kernel_size=3, padding=1)
        
        # Enhanced time embedding projection (512-dimensional instead of 256)
        self.t_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )
        
        # All blocks now use improved time conditioning
        self.down2 = DownBlock(256, 384, 512, down_sample=False, num_heads=16, attn=False, norm_channels=8)
        self.down3 = DownBlock(384, 512, 512, down_sample=True, num_heads=16, attn=True, norm_channels=8)
        
        self.up2 = UpBlockUnet(512, 384, 512, up_sample=True, num_heads=16, attn=True, norm_channels=8)
        self.up3 = UpBlockUnet(768, 256, 512, up_sample=False, num_heads=16, attn=False, norm_channels=8)
        
        # Final output layers with time conditioning
        self.norm_out = nn.GroupNorm(8, 512)
        self.conv_out = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        

    def forward(self, x, t, encoder_skips=None):
        
        
        
        enc_sk1, enc_sk2, enc_sk3 = None, None, None
        if encoder_skips is not None:
            if len(encoder_skips) > 0:
                enc_sk0 = encoder_skips[0]
            if len(encoder_skips) > 1:
                enc_sk1 = encoder_skips[1]
            if len(encoder_skips) > 2:
                enc_sk2 = encoder_skips[2]
            if len(encoder_skips) > 3:
                enc_sk3 = encoder_skips[3]    
        #print(enc_sk0.shape)
        x = torch.cat([x, enc_sk0], dim=1)
        out = self.conv_in(x)

        # Enhanced time embedding (512-dimensional)
        t_emb = get_time_embedding(torch.as_tensor(t).float(), 512)  # Use float instead of long
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        down_outs.append(out)
        
        assert out.shape == enc_sk1.shape
        out = out + enc_sk1
        
        # All forward passes now include time embedding
        out = self.down2(out, t_emb)
        down_outs.append(out)
        assert out.shape == enc_sk2.shape
        out = out + enc_sk2

        out = self.down3(out, t_emb)
        
        # Upsample blocks with time conditioning
        assert out.shape == enc_sk3.shape
        out = out + enc_sk3
        out = self.up2(out, None, t_emb)
        out = self.up3(out, down_outs.pop(), t_emb)

        out = torch.cat([out, down_outs.pop()], dim=1)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        
        return out
    
if __name__ == "__main__":

    # Assume your model classes (Encoder, Unet, etc.) are already defined above

    # Create dummy input
    dummy_image = torch.randn(1, 3, 256, 256)     # Batch size 1, 3 channels, 256x256
    dummy_latent = torch.randn(1, 8, 32, 32)      # Latent input for Unet
    dummy_time = torch.tensor([0.5])             # Normalized timestep in [0, 1]

    # Instantiate models
    encoder = Encoder(im_channels=3)
    unet = Unet(im_channels=8)

    # Forward pass through encoder
    encoder_skips = encoder(dummy_image)

    # Forward pass through unet
    output = unet(dummy_latent, dummy_time, encoder_skips)

    # Print output shape
    print("Output shape from Unet:", output.shape)
            

