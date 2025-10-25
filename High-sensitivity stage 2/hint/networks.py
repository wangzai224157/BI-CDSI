
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import  numpy as np
from typing  import  Tuple

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

############# HINT
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Feed-Forward Network (FFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)  


        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Squeeze-and-Channel Attention Layer (SCAL)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        #### Channel branch
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        
        #### Spatial branch
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)
        ##########

    def forward(self, x):
        #### Channel branch
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        #### Spatial branch
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.upsample(y)
        out = y * out
        ###########

        out = self.project_out(out)
        return out


##########################################################################
########### Sandwich Block
class SandwichBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SandwichBlock, self).__init__()
        
        
        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        
        self.norm1 = LayerNorm(dim, LayerNorm_type)

        self.attn = Attention(dim, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        
        x = x + self.ffn1(self.norm1_1(x))
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
        




##########################################################################
## Gated Embedding layer
class GatedEmb(nn.Module):
    def __init__(self, in_c = 12, embed_dim =48, bias=False):
        super(GatedEmb, self).__init__()

        self.gproj1 = nn.Conv2d(in_c, embed_dim*2, kernel_size=3,stride=1,padding=1,bias=bias)


    def forward(self, x):
        #x = self.proj(x)
        x = self.gproj1(x)
        x1, x2 = x.chunk(2, dim=1)

        x = F.gelu(x1) * x2

        return x


##########################################################################
## Mask-aware Pixel-Shuffle Down-Sampling (MPD)
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

        self.body2 = nn.Sequential(nn.PixelUnshuffle(2)) 

        
        self.proj = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=1, padding=1, groups=n_feat * 2, bias=False)
    def forward(self, x):

        out = self.body(x)
        #out_mask = self.body2(mask)
        b,n,h,w = out.shape
        t = torch.zeros((b,2*n,h,w)).cuda()
        for i in range(n):
            t[:,2*i,:,:] = out[:,i,:,:]
        #for i in range(n):
            #if i <= 3:
                #t[:,2*i+1,:,:] = out_mask[:,i,:,:]
            #else:
                #t[:,2*i+1,:,:] = out_mask[:,(i%4),:,:]

        return self.proj(t)



class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
	
    def forward(self, x):
        return self.body(x)


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    
    # Convert mask to uint8 type if it's not already
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Convert boolean to integer for bitwise operations
    correct_holes_int = 1 if correct_holes else 0
    
    working_mask = np.bitwise_xor(mask, correct_holes_int).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True



def postprocess_masks(self, mask_output):
        """
        对所有mask应用小区域去除处理
        
        参数:
            mask_output: 模型输出的mask，形状为(B, C, H, W)
            
        返回:
            processed_mask: 处理后的mask，形状与输入相同
        """
        batch_size, channels, height, width = mask_output.shape
        
        # 确保mask是二值的 (0或1)
        mask_binary = (mask_output > 0.5).float()
        
        processed_mask = torch.zeros_like(mask_output)
        
        # 对每个batch中的每个mask进行处理
        for b in range(batch_size):
            for c in range(channels):
                # 转换为numpy数组
                mask_np = mask_binary[b, c].cpu().numpy()
                
                # 如果mask不为空，则处理
                if np.sum(mask_np) > 0:
                    # 转换为uint8类型，因为remove_small_regions需要
                    mask_uint8 = (mask_np * 255).astype(np.uint8)
                    
                    # 复制mask用于处理
                    processed_mask_np, changed = self.remove_small_regions(
                        mask_uint8, area_thresh=50, mode="islands"
                    )
                    
                    # 将处理后的mask转换回torch.Tensor
                    processed_mask[b, c] = torch.from_numpy(processed_mask_np / 255).float().to(mask_output.device)
        
        return processed_mask


##########################################################################
##---------- HINT -----------------------
class HINT(nn.Module):
    def __init__(self,
                 inp_channels=12,
                 out_channels=3,
                 dim=48,
                 #num_blocks=[4, 6, 6, 8],
                 num_blocks=[2, 3, 3, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 use_mask_head=True,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 mask_out_channels = 1
                 ):

        super(HINT, self).__init__()
        self.use_mask_head = use_mask_head
        self.patch_embed = GatedEmb(in_c=inp_channels, embed_dim=dim)

        self.encoder_level1 = nn.Sequential(*[
            SandwichBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.output = nn.Sequential(nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                                    )     
        self.outputmask = nn.Sequential(nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                                    )
        if use_mask_head:
            # 直接在HINT中添加mask检测头，使用decoder_level1的输出作为输入
            self.mask_head = YOLOMaskHead(
                in_channels=int(dim * 2 ** 1),  # 使用decoder的输出通道数
                num_classes=mask_out_channels
            )
        else:
            self.mask_head = None


    def forward(self, inp_img,return_mask = True):

        """"
        np_img torch.Size([1, 3, 256, 256])
        mask_whole torch.Size([1, 1, 256, 256])
        mask_half  torch.Size([1, 1, 128, 128])
        mask_quarter
        torch.Size([1, 1, 64, 64])
        mask_tiny
        torch.Size([1, 1, 32, 32])
        """
        
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)

        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1_mid = self.decoder_level1(inp_dec_level1)
        
        out_mask = self.outputmask(out_dec_level1_mid)

        out_mask = (torch.tanh(out_mask) + 1) / 2
        if self.use_mask_head and return_mask:

            mask_output = self.mask_head(out_dec_level1_mid)
            


            mask_np = mask_output.detach().cpu().numpy()
            cleaned_mask = np.zeros_like(mask_np)
            for i in range(mask_np.shape[0]):  # For each sample in batch
                for j in range(mask_np.shape[1]):  # For each channel
                    cleaned_mask_2d, _ = remove_small_regions(mask_np[i, j], area_thresh=5.0, mode='holes')
                    cleaned_mask_2d, _ = remove_small_regions(mask_np[i, j], area_thresh=5.0, mode='islands')
                    cleaned_mask[i, j] = cleaned_mask_2d
            mask_output_cleaned = torch.from_numpy(cleaned_mask).float().to(inp_img.device)



            return mask_output
        else:
            return out_mask




        
     
class HINT1(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 #num_blocks=[4, 6, 6, 8],
                 num_blocks=[2, 3, 3, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):

        super(HINT1, self).__init__()

        self.patch_embed = GatedEmb(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            SandwichBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.output = nn.Sequential(nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                                    )        


    def forward(self, inp_img):

        """"
        np_img torch.Size([1, 3, 256, 256])
        mask_whole torch.Size([1, 1, 256, 256])
        mask_half  torch.Size([1, 1, 128, 128])
        mask_quarter
        torch.Size([1, 1, 64, 64])
        mask_tiny
        torch.Size([1, 1, 32, 32])
        """
        print(inp_img.shape)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)

        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.output(out_dec_level1)

        out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2
        return out_dec_level1
        


class YOLOMaskHead(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        """简化版YOLOMaskHead，处理单个特征图"""
        super(YOLOMaskHead, self).__init__()
        self.num_classes = num_classes
        
        # 调整通道数
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 最终的mask预测头
        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, feat):
        fused_feat = self.fusion_layer(feat)
        mask_pred = self.mask_head(fused_feat)
        return mask_pred



"""
#以下是使用钩子并保存特征图的，但是出现了内存不足的问题。
        
class YOLOMaskHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1, mask_size=64):
        #简化版YOLOMaskHead，处理单个特征图
        super(YOLOMaskHead, self).__init__()
        self.num_classes = num_classes
        self.mask_size = mask_size
        
        # 调整通道数
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 最终的mask预测头
        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, feat):
        # 确保输入是单个特征图
        if isinstance(feat, list):
            feat = feat[0]  # 取第一个特征图
        
        fused_feat = self.fusion_layer(feat)
        mask_pred = self.mask_head(fused_feat)
        return mask_pred

class HINTWithMask(HINT):
    def __init__(self, 
                 in_channels=12,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 mask_head_channels=256,
                 mask_size=64,
                 num_classes=1
                 ):
        super().__init__(
            inp_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )
        
        self.save_features = True
        self.mask_head = YOLOMaskHead(
            in_channels_list=[mask_head_channels],
            num_classes=num_classes,
            mask_size=mask_size
        )
        
        # 添加中间特征提取钩子（假设HINT模型生成多尺度特征）
        self.feature_maps = []
        self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        #注册钩子以获取中间特征图
        def hook(module, input, output):
            self.feature_maps.append(output)
        
        # 示例：在编码器的不同层级注册钩子
        hooks = [
            self.encoder_level1,
            self.encoder_level2,
            self.encoder_level3
        ]
        for module in hooks:
            module.register_forward_hook(hook)
    
    def forward(self, inp_img, return_mask=True):
        self.feature_maps = []  # 清空之前的特征图
        mask_output = super().forward(inp_img)
        
        if return_mask and self.mask_head:
            # 确保feature_maps是一个列表（即使只有一个元素）
            if not isinstance(self.feature_maps, list) or len(self.feature_maps) == 0:
                # 如果没有提取到特征图，使用mask_output创建一个临时列表
                feats = [mask_output]
            else:
                feats = self.feature_maps
            
            # 调整特征图尺寸以匹配YOLOMaskHead的期望
            adjusted_feats = []
            for i, feat in enumerate(feats):
                # 假设需要调整到特定尺寸，这里以双线性插值为例
                adjusted_feat = torch.nn.functional.interpolate(
                    feat, 
                    size=(mask_output.shape[2], mask_output.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
                adjusted_feats.append(adjusted_feat)
            
            final_mask_output = self.mask_head(adjusted_feats)
            return final_mask_output
        else:
            return mask_output



class HINTWithMask(HINT):
    def __init__(self, 
                 inp_channels=12,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 mask_head_channels=3,  # 用于mask头的通道数
                 mask_size=64,           # mask的大小
                 num_classes=1           # 类别数量，这里设置为1用于检测马赛克
                 ):
      
        #HINT模型与YOLOMaskHead的结合
        
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )
        
        # 保存HINT模型的中间特征图
        self.save_features = True
        
        # 添加mask检测头
        self.mask_head = YOLOMaskHead(
            in_channels=mask_head_channels,
            num_classes=num_classes,
            mask_size=mask_size
        )

    def forward(self, inp_img, return_mask=True):
        # 前向传播HINT模型
        mask_output = super().forward(inp_img)  # 只接收一个返回值
        
        # 如果需要返回mask，通过mask_head
        if return_mask and self.mask_head:
            # 使用HINT模型的输出作为mask_head的输入
            final_mask_output = self.mask_head(mask_output)
            return final_mask_output
        else:
            return mask_output  # 或者根据需求返回其他内容
"""