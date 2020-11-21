from typing import List

import torch
from forknet.modules import EncodeModule, DecodeModule, Concat, ConvModule


class DecoderTrack(torch.nn.Module):
    def __init__(self):
        super(DecoderTrack, self).__init__()
        self.decoders = torch.nn.ModuleList([
            DecodeModule(2 ** (j + 2), 1 if j == 1 else 2 ** (j + 1)) for j in range(5, 0, -1)
        ])
        self.concat = Concat()
        self.conv_modules = torch.nn.ModuleList([
            ConvModule(2 ** (j + 3), 2 ** (j + 2)) for j in range(5, 0, -1)
        ])

    def decoder_track(self, x: List[torch.Tensor]) -> torch.Tensor:
        inp = x.pop()
        for conv, decoder in zip(self.conv_modules, self.decoders):
            inp = decoder(conv(self.concat(inp, x.pop())))
            print(inp.shape)
        return inp

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.decoder_track(x)


class ForkNet(torch.nn.Module):
    def __init__(self, classes):
        super(ForkNet, self).__init__()
        self.encoders = torch.nn.ModuleList([
            EncodeModule(1 if i == 1 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(1, 7)
        ])
        self.base_decoder = DecodeModule(256, 128)
        self.decoder_track = DecoderTrack()
        self.classes = classes

    def encoder_track(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for encoder in self.encoders:
            x = encoder(x)
            out.append(x)
            print(x.shape)
        out.append(self.base_decoder(out.pop()))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_track(self.encoder_track(x))
