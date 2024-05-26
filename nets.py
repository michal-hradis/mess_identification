import torch
import json
import logging
from nets_pretrained import PoolingDecoder, ConvDecoder, PretrainedEncoder, EmbeddingModel
from nets_att_decoders import AttDecoder

def net_factory(encoder_config, decoder_config, emb_dim, normalize=True):
    if type(encoder_config) == str:
        encoder_config = json.loads(encoder_config)
    if type(decoder_config) == str:
        decoder_config = json.loads(decoder_config)

    if encoder_config['type'].lower() == 'sm':
        del encoder_config['type']
        encoder = PretrainedEncoder(in_channels=3, **encoder_config)
    else:
        logging.error(f'Unknown encoder type "{encoder_config["type"]}"')
        exit(-1)

    data = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    features = encoder(data)
    for i, f in enumerate(features):
        print(f'Features {i} {f.shape}')
    feature_dim = features[-1].shape[1]

    if decoder_config['type'].lower() == 'avg_pool':
        del decoder_config['type']
        decoder = PoolingDecoder(in_channels=feature_dim, emb_dim=emb_dim, **decoder_config)
    elif decoder_config['type'].lower() == 'conv':
        del decoder_config['type']
        decoder = ConvDecoder(in_channels=feature_dim *4*4, emb_dim=emb_dim, **decoder_config)
    elif decoder_config['type'].lower() == 'attention':
        del decoder_config['type']
        decoder = AttDecoder(in_channels=feature_dim, emb_dim=emb_dim, **decoder_config)
    else:
        logging.error(f'Unknown decoder type "{decoder_config["type"]}"')
        exit(-1)

    model = EmbeddingModel(encoder, decoder, normalize=normalize)
    return model

def create_vgg_block(input_channels, output_channels, subsampling=(2, 2), layer_count=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        torch.nn.LeakyReLU(),
        # torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        # torch.nn.LeakyReLU(),
        torch.nn.MaxPool2d(kernel_size=subsampling, stride=subsampling),
        torch.nn.InstanceNorm2d(num_features=output_channels)
    )




class Encoder(torch.nn.Module):
    def __init__(self, dim=256, base_channels=32):
        super().__init__()
        self.conv = torch.nn.Sequential(
            create_vgg_block(3, base_channels, subsampling=(2, 2)),
            create_vgg_block(base_channels, base_channels * 2, subsampling=(2, 2)),
            create_vgg_block(base_channels * 2, base_channels * 4, subsampling=(2, 2)),
            create_vgg_block(base_channels * 4, base_channels * 8, subsampling=(2, 2)),
        )
        self.attention_dim = base_channels * 8

        self.pos_enc = PositionalEncoding2D(d_model=self.attention_dim, max_len=10)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.attention_dim,
                                                         dim_feedforward=self.attention_dim * 2, nhead=4)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        # self.output_layer = torch.nn.Sequential(torch.nn.Linear(self.attention_dim, dim), torch.nn.LeakyReLU())
        output_layer = torch.nn.Linear(self.attention_dim, dim, bias=False)
        with torch.no_grad():
            output_layer.weight.data *= 0.1

        self.output_layer = torch.nn.Sequential(torch.nn.LeakyReLU(), torch.nn.LayerNorm(self.attention_dim),
                                                output_layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.pos_enc(x)
        x = x.reshape(x.shape[0], self.attention_dim, -1)
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1)
        x = x[:, :, 0]
        x = self.output_layer(x)
        # x = torch.nn.functional.normalize(x)
        return x

#class Resne
