import math
import torch


class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        coordinates = (position * div_term).permute(1, 0)
        pe = torch.zeros(1, d_model, max_len, 1)
        pe[0, 0::2, :, 0] = torch.sin(coordinates)
        pe[0, 1::2, :, 0] = torch.cos(coordinates)

        pe2 = torch.zeros(1, d_model, 1, max_len)
        pe2[0, 0::2, 0, :] = torch.sin(coordinates)
        pe2[0, 1::2, 0, :] = torch.cos(coordinates)
        pe2[...] = torch.flip(pe2, dims=[1])

        self.register_buffer('pe', pe)
        self.register_buffer('pe2', pe2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:, :, :x.size(2), :] + self.pe2[:, :, :, :x.size(3)]


class DecoderBlock(torch.nn.Module):
    def __init__(self, input_channels, dim=1024, size=8, num_layers=4, heads=8):
        super().__init__()
        self.dim = dim
        self.size = size
        self.heads = heads
        self.input_transform = torch.nn.Sequential(torch.nn.LeakyReLU(),
                                                   torch.nn.Conv1d(input_channels, dim, kernel_size=1))
        self.start_transform = torch.nn.Sequential(torch.nn.LeakyReLU(), torch.nn.Linear(input_channels, dim * size))
        layer = torch.nn.TransformerDecoderLayer(dim, self.heads, dim_feedforward=dim * 2, dropout=0.1,
                                                 batch_first=False, norm_first=False)
        self.blocks = torch.nn.TransformerDecoder(layer, num_layers)

    def forward(self, x):
        sum_x = torch.nn.functional.adaptive_avg_pool1d(x, 1)[:, :, 0]
        init = self.start_transform(sum_x).reshape(-1, self.dim, self.size)
        x = self.input_transform(x)
        x = x.permute(2, 0, 1)
        init = init.permute(2, 0, 1)
        x = self.blocks(init, x)
        x = x.permute(1, 2, 0)
        return x[:, :, 0]


class AttDecoder(torch.nn.Module):
    def __init__(self, in_channels, emb_dim, decoder_dim=512):
        super().__init__()

        self.attention_dim = in_channels
        self.decoder_dim = decoder_dim

        self.pos_enc = PositionalEncoding2D(d_model=self.attention_dim, max_len=10)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.attention_dim,
                                                         dim_feedforward=self.attention_dim * 2, nhead=4)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.decoder = DecoderBlock(self.attention_dim, self.decoder_dim)

        output_layer = torch.nn.Linear(self.decoder_dim, emb_dim, bias=False)
        with torch.no_grad():
            output_layer.weight.data *= 0.1
        self.output_layer = torch.nn.Sequential(torch.nn.LeakyReLU(), torch.nn.LayerNorm(self.decoder_dim), output_layer)

    def forward(self, x):
        x = self.pos_enc(x)
        x = x.reshape(x.shape[0], self.attention_dim, -1)
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = self.decoder(x)
        x = self.output_layer(x)
        return x
