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
    def __init__(self, in_channels: int, emb_dim: int, operation: str = "avg", expansion: int = -1,
                 normalize: bool = True, activation: bool = True, dropout: float = 0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.expansion = expansion
        self.normalize = normalize
        if operation not in ["avg", "max"]:
            raise ValueError(f"Unknown pooling operation {operation}. Possible values are 'avg' and 'max'")

        output_layers = []
        if dropout > 0:
            output_layers.append(torch.nn.Dropout(dropout))

        if self.expansion > 0:
            output_layers.append(torch.nn.Conv2d(in_channels, self.expansion, 1))
            in_channels = self.expansion

        if operation == "avg":
            output_layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        elif operation == "max":
            output_layers.append(torch.nn.AdaptiveMaxPool2d((1, 1)))
        output_layers.append(torch.nn.Flatten())

        if activation:
            output_layers.append(torch.nn.LeakyReLU())

        if self.normalize:
            output_layers.append(torch.nn.LayerNorm(in_channels))

        output_layers.append(torch.nn.Linear(in_channels, emb_dim, bias=False))
        with torch.no_grad():
            output_layers[-1].weight.data *= 0.1

        self.output_layer = torch.nn.Sequential(*output_layers)

    def forward(self, x):
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
    def __init__(self, encoder, decoder, normalize=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.normalize = normalize

    def forward(self, x):
        # input is uint8 [0, 255], we need to convert it to float and normalize
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.encoder(x)[-1]
        embeddings = self.decoder(x)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

