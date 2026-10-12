"""DeepJSCC codec (from DiffJSCC) -- image over a noisy channel -> MLIR.

    python workloads/capture.py deepjscc --formats fp32,int8
    <venv>/bin/python workloads/capture_consistent.py deepjscc int8 <bundle_dir>

Capture unit: the JSCC **encoder + decoder** -- a reflection-padded conv stem, strided
downsampling convs, ResNet blocks with SNR-conditioned feature modulation, a channel
projection, then ConvTranspose upsampling back to an RGB image. This is the deployable codec.

It is NOT the full DiffJSCC pipeline. That adds a Stable-Diffusion-2.1 UNet, an AutoencoderKL
and an OpenCLIP text encoder driven for tens of sampling steps; the paper's reconstruction
quality comes from that diffusion stage, so do not quote those numbers for this bundle.

The AWGN channel itself is excluded: it multiplies in fresh Gaussian noise, which would make
the golden unreproducible. What is captured is encode -> (power normalize) -> decode, which is
the deterministic compute; the channel is a runtime input in the real system.

Env:
    DIFFJSCC_DIR   upstream checkout (default: /path/to/DiffJSCC)
    M2M_JSCC_SIZE  square input resolution (default: 64; the paper trains at 256, which makes
                   the im2col intermediates large -- raise deliberately, not by habit)
    M2M_JSCC_NGF   base width (default: 16; upstream uses 64)

Upstream: https://github.com/mingyuyng/DiffJSCC  (model/deepjscc_cnn.py, configs/model/deepjscc_cnn.yaml)
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
from torch import nn

_UPSTREAM = Path(os.environ.get("DIFFJSCC_DIR", "/path/to/DiffJSCC"))


def _stub_training_deps() -> None:
    """Stub the training-only imports ``deepjscc_cnn.py`` performs at module scope.

    The file imports pytorch_lightning, LPIPS/PSNR metrics, pytorch_msssim and einops for its
    LightningModule. The encoder and decoder we capture are plain ``nn.Module``s, so stubbing is
    honest and avoids pulling a training stack into a capture venv.
    """
    for name in ("pytorch_lightning", "pytorch_lightning.utilities",
                 "pytorch_lightning.utilities.types", "pytorch_msssim",
                 "utils", "utils.metrics", "einops"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["pytorch_lightning"].LightningModule = nn.Module
    sys.modules["pytorch_lightning.utilities.types"].STEP_OUTPUT = object
    for fn in ("ssim", "ms_ssim", "SSIM", "MS_SSIM"):
        setattr(sys.modules["pytorch_msssim"], fn, lambda *a, **k: None)
    sys.modules["utils.metrics"].calculate_psnr_pt = lambda *a, **k: None
    sys.modules["utils.metrics"].LPIPS = object
    sys.modules["einops"].rearrange = lambda *a, **k: None


class _Codec(nn.Module):
    """encode -> power-normalize -> decode. Deterministic; the channel noise stays outside."""

    def __init__(self, enc: nn.Module, dec: nn.Module) -> None:
        super().__init__()
        self.enc, self.dec = enc, dec

    def forward(self, image: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        latent = self.enc(image, csi)
        return self.dec(latent, csi)


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    _stub_training_deps()
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    from model.deepjscc_cnn import define_Decoder, define_Encoder  # type: ignore

    size = int(os.environ.get("M2M_JSCC_SIZE", 64))
    ngf = int(os.environ.get("M2M_JSCC_NGF", 16))
    # configs/model/deepjscc_cnn.yaml: norm='batch', C_channel=4, AWGN (so C_extend=1).
    enc = define_Encoder(input_nc=3, ngf=ngf, max_ngf=4 * ngf, C_channel=4, C_extend=1,
                         n_blocks=1, n_downsample=2, norm="batch")
    dec = define_Decoder(output_nc=3, ngf=ngf, max_ngf=4 * ngf, n_downsample=2, C_channel=4,
                         C_extend=1, n_blocks=1, norm="batch")
    print("[deepjscc] RANDOM INIT — upstream publishes only full DiffJSCC checkpoints "
          "(Stable-Diffusion-merged, several GB); the golden checks lowering, not rate-distortion",
          file=sys.stderr)

    image = torch.rand(1, 3, size, size)          # RGB in [0, 1], as the sigmoid output is
    csi = torch.full((1, 1), 10.0)                # channel SNR in dB
    return _Codec(enc.eval(), dec.eval()).eval(), (image, csi)
