import torch
from torch import nn
from attention import RowAttention, ColumnAttention
        

class Encoder(nn.Module):
    """ Encoder_summary

    Args:
    """    
    def __init__(
        self,
        embed_size: int,
    ) -> None:
        super(Encoder, self).__init__()
        
        self.row_attn = RowAttention(embed_size)
        self.column_attn = ColumnAttention(embed_size)
        
    def forward(self, x):
        out = self.row_attn(x)
        out = self.column_attn(out)
        
        return out


class GrayScaleEncoder(nn.Module):
    """ GrayScaleEncoder_summary

    Args:

    """
    def __init__(
        self,
        in_embed: int, 
        out_embed: int,
        n_layers: int = 4
    ) -> None:
        super(GrayScaleEncoder, self).__init__()
        
        self.x_g_embed = nn.Embedding(in_embed, out_embed)
        self.enc_layers = nn.Sequential(
            *[Encoder(out_embed) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(out_embed, out_embed)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, gray_images):
        
        gray_embedding = self.x_g_embed(gray_images)
        gray_embedding = self.enc_layers(gray_embedding)
        p_c = self.linear(gray_embedding)
        p_c = self.softmax(p_c)
        
        return p_c, gray_embedding
        

class InnerDecoder(nn.Module):
    """ InnerDecoder_summary

    Args:
        
    """
    def __init__(
        self,
        in_embed: int, 
        out_embed: int,
        n_layers: int = 4
    ) -> None:
        super(InnerDecoder, self).__init__()
    
    def forward(self, embedding_images):
        pass


class OuterDecoder(nn.Module):
    """ OuterDecoder_summary

    Args:
        
    """
    def __init__(
        self,
        in_embed: int, 
        out_embed: int,
        n_layers: int = 4
    ) -> None:
        super(OuterDecoder, self).__init__()
        self.x_s_c_embed = nn.Embedding(in_embed, out_embed)
        
    def forward(self, color_images, gray_embedding):
        pass


class ColTranCore(nn.Module):
    """ ColTranCore_summary

    Args:
        
    """
    def __init__(
        self, 
        in_embed: int, 
        out_embed: int,
        n_layers: int = 4
    ) -> None:
        super(ColTranCore, self).__init__()
        
        self.gray_encoder = GrayScaleEncoder(in_embed, out_embed, n_layers)
        
        self.linear = nn.Linear()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        pass
    
    
if __name__ == "__main__":
    
    gray_image = torch.randint(0, 255, (2, 64, 64))
    gray_enc = GrayScaleEncoder(256+1, 512)    
    p_c, gray_embedding = gray_enc(gray_image)
    
    print("p_c : ", p_c.shape)
    print("gray_embedding : ", gray_embedding.shape)
    
    