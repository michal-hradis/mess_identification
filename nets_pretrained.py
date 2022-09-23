from typing import Optional, Union
import torch
from segmentation_models_pytorch.encoders import get_encoder


class PretrainedEncoder(torch.nn.Module):
    def __init__(self, name: str = "resnet34", depth: int = 5,
                 weights: Optional[str] = "imagenet", in_channels: int = 3
    ):
        super().__init__()
        self.encoder = get_encoder(
            name,
            in_channels=in_channels,
            depth=depth,
            weights=weights)

    def forward(self, x):
        features = self.encoder(x)
        return features


class PoolingDecoder(torch.nn.Module):
    def __init__(self, in_channels: int, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        output_layer = torch.nn.Linear(in_channels, emb_dim, bias=False)
        with torch.no_grad():
            output_layer.weight.data *= 0.1
        self.output_layer = torch.nn.Sequential(torch.nn.LeakyReLU(), torch.nn.LayerNorm(in_channels),
                                                output_layer)

    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))[:, :, 0, 0]
        x = self.output_layer(x)
        return x


class ConvDecoder(torch.nn.Module):
    def __init__(self, in_channels: int, inner_channels: int,  emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        output_layer = torch.nn.Linear(inner_channels, emb_dim, bias=False)
        with torch.no_grad():
            output_layer.weight.data *= 0.1
        self.output_layer = torch.nn.Sequential( torch.nn.Linear(in_channels, inner_channels), torch.nn.LeakyReLU(), torch.nn.LayerNorm(inner_channels),
                                                output_layer)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.output_layer(x)
        return x


class EmbeddingModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.decoder(x)
        return x

