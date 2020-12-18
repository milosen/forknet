from typing import List, Tuple

import torch
from forknet.modules import EncodeModule, DecodeModule, ConvModule, Map


class DecoderTrack(torch.nn.Module):
    def __init__(self):
        super(DecoderTrack, self).__init__()
        self.decoders = torch.nn.ModuleList([
            DecodeModule(2 ** (j + 2), 1 if j == 1 else 2 ** (j + 1)) for j in range(5, 0, -1)
        ])
        self.conv_modules = torch.nn.ModuleList([
            ConvModule(2 ** (j + 3), 2 ** (j + 2)) for j in range(5, 0, -1)
        ])

    def decoder_track(self, x: List[torch.Tensor]) -> torch.Tensor:
        inp = x.pop()
        for conv, decoder in zip(self.conv_modules, self.decoders):
            inp = decoder(conv(torch.cat([inp, x.pop()], dim=1)))
        return inp

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.decoder_track(x)


class ForkNet(torch.nn.Module):
    def __init__(self, n_classes):
        super(ForkNet, self).__init__()
        self.n_classes = n_classes
        self.encoders = torch.nn.ModuleList([
            EncodeModule(1 if i == 1 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(1, 7)
        ])
        self.base_decoder = DecodeModule(256, 128)
        self.decoder_tracks = torch.nn.ModuleList([
            DecoderTrack() for _ in range(self.n_classes)
        ])
        self.maps = torch.nn.ModuleList([
            Map(in_channels=1, out_channels=1) for _ in range(self.n_classes)
        ])

    def encoder_track(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []  # TODO more efficient to concat the tensors?
        for encoder in self.encoders:
            x = encoder(x)
            out.append(x)
        out.append(self.base_decoder(out.pop()))
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(
            map_module(decoder_track(self.encoder_track(x)))
            for decoder_track, map_module in zip(self.decoder_tracks, self.maps)
        )
