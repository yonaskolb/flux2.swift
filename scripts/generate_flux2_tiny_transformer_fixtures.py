#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2 transformer fixtures for numerical parity checks."
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_tiny",
        help="Output directory for the tiny transformer snapshot and fixtures.",
    )
    parser.add_argument(
        "--diffusers-path",
        default="temp/diffusers/src",
        help="Path to diffusers src/ for the reference implementation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for weights and fixtures.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for running the reference model.",
    )
    parser.add_argument("--text-seq-len", type=int, default=3, help="Text token sequence length.")
    parser.add_argument("--height", type=int, default=2, help="Latent height.")
    parser.add_argument("--width", type=int, default=2, help="Latent width.")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def prepare_text_ids(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(1, device=device)
    h = torch.arange(1, device=device)
    w = torch.arange(1, device=device)
    l = torch.arange(seq_len, device=device)
    coords = torch.cartesian_prod(t, h, w, l)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def prepare_latent_ids(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(1, device=device)
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    l = torch.arange(1, device=device)
    coords = torch.cartesian_prod(t, h, w, l)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def main() -> None:
    args = parse_args()
    diffusers_src = Path(args.diffusers_path)
    if diffusers_src.exists():
        sys.path.insert(0, str(diffusers_src))
    else:
        raise SystemExit(f"diffusers path not found: {diffusers_src}")

    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = {
        "patch_size": 1,
        "in_channels": 8,
        "out_channels": 8,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 1,
        "joint_attention_dim": 8,
        "timestep_guidance_channels": 4,
        "mlp_ratio": 2.0,
        "axes_dims_rope": [2, 2, 2, 2],
        "rope_theta": 2000,
        "eps": 1e-6,
        "guidance_embeds": True,
    }

    model = Flux2Transformer2DModel(**config).to(device=device, dtype=dtype)
    model.eval()

    batch = 1
    image_seq_len = args.height * args.width
    hidden_states = torch.randn(
        (batch, image_seq_len, config["in_channels"]), device=device, dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        (batch, args.text_seq_len, config["joint_attention_dim"]), device=device, dtype=dtype
    )
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    img_ids = prepare_latent_ids(batch, args.height, args.width, device=device)
    txt_ids = prepare_text_ids(batch, args.text_seq_len, device=device)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )[0]

    out_dir = Path(args.out_dir)
    transformer_dir = out_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    with (transformer_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    weights = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(weights, str(transformer_dir / "model.safetensors"))

    inputs = {
        "hidden_states": hidden_states.cpu(),
        "encoder_hidden_states": encoder_hidden_states.cpu(),
        "timestep": timestep.cpu(),
        "img_ids": img_ids.cpu(),
        "txt_ids": txt_ids.cpu(),
    }
    save_file(inputs, str(out_dir / "transformer_inputs.safetensors"))

    expected = {"output": output.cpu()}
    save_file(expected, str(out_dir / "transformer_expected.safetensors"))

    print(f"Wrote tiny transformer snapshot to {transformer_dir}")
    print(f"Wrote inputs to {out_dir / 'transformer_inputs.safetensors'}")
    print(f"Wrote expected output to {out_dir / 'transformer_expected.safetensors'}")


if __name__ == "__main__":
    main()
