import math
import torch
from torch import nn


class MLP(nn.Module):
    """ MLP_summary

    Args:
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int
    ) -> None:
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


class ColumnAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()

        self.dim = embed_dim

        self.Q1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V1 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.Q2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V2 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.Q3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V3 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.Q4 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K4 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V4 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.layernorm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(n_heads*embed_dim, embed_dim)
        self.mlp = MLP(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        print(images.shape)
        batch, rows, cols, _ = images.shape
        out = torch.empty_like(images)

        for col in range(cols):
            layernorm_images = self.layernorm(images[:, :, col, :])
            attn1 = self.softmax(torch.matmul(self.Q1(layernorm_images),
                                              self.K1(layernorm_images).transpose(1, 2)) / math.sqrt(self.dim))
            attn2 = self.softmax(torch.matmul(self.Q2(layernorm_images),
                                              self.K2(layernorm_images).transpose(1, 2)) / math.sqrt(self.dim))
            attn3 = self.softmax(torch.matmul(self.Q3(layernorm_images),
                                              self.K3(layernorm_images).transpose(1, 2)) / math.sqrt(self.dim))
            attn4 = self.softmax(torch.matmul(self.Q4(layernorm_images),
                                              self.K4(layernorm_images).transpose(1, 2)) / math.sqrt(self.dim))
            # Ai : batch * H * H
            attn1 = torch.matmul(attn1, self.V2(layernorm_images))
            attn2 = torch.matmul(attn2, self.V2(layernorm_images))
            attn3 = torch.matmul(attn3, self.V3(layernorm_images))
            attn4 = torch.matmul(attn4, self.V4(layernorm_images))

            multi_self_attn = self.linear(
                torch.cat([attn1, attn2, attn3, attn4], dim=2))
            multi_self_attn += images[:, :, col, :]
            out[:, :, col, :] = self.mlp(
                self.layernorm(multi_self_attn)) + multi_self_attn

        return out


class RowAttention(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()

        self.dim = embed_dim
        self.Q1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V1 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.Q2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V2 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.Q3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V3 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.Q4 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K4 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V4 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.layernorm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(n_heads*embed_dim, embed_dim)
        self.mlp = MLP(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        batch, rows, col, _ = images.shape
        out = torch.empty_like(images)

        for row in range(rows):
            layernorm_images = self.layernorm(images[:, row, :, :])
            attn1 = self.softmax(torch.matmul(self.Q1(layernorm_images),
                                              self.K1(layernorm_images).transpose(1, 2)) / math.sqrt(self.dim))
            attn2 = self.softmax(torch.matmul(self.Q2(layernorm_images),
                                              self.K2(layernorm_images).transpose(1, 2)) / math.sqrt(self.dim))
            attn3 = self.softmax(torch.matmul(self.Q3(layernorm_images),
                                              self.K3(layernorm_images).transpose(1, 2)) / math.sqrt(self.dim))
            attn4 = self.softmax(torch.matmul(self.Q4(layernorm_images),
                                              self.K4(layernorm_images).transpose(1, 2)) / math.sqrt(self.dim))
            # Ai : batch * H * H
            attn1 = torch.matmul(attn1, self.V2(layernorm_images))
            attn2 = torch.matmul(attn2, self.V2(layernorm_images))
            attn3 = torch.matmul(attn3, self.V3(layernorm_images))
            attn4 = torch.matmul(attn4, self.V4(layernorm_images))

            multi_self_attn = self.linear(
                torch.cat([attn1, attn2, attn3, attn4], dim=2))
            multi_self_attn += images[:, row, :, :]
            out[:, row, :, :] = self.mlp(
                self.layernorm(multi_self_attn)) + multi_self_attn

        return out
