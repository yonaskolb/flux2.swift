#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2 position/packing fixtures for numerical parity checks."
    )
    parser.add_argument(
        "--snapshot-dir",
        default="fixtures/flux2_tiny",
        help="Directory containing the tiny fixtures.",
    )
    parser.add_argument(
        "--image-scale",
        type=int,
        default=10,
        help="Scale used for image id time offsets.",
    )
    return parser.parse_args()


def prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    t = torch.arange(1, device=x.device)
    h = torch.arange(1, device=x.device)
    w = torch.arange(1, device=x.device)
    l = torch.arange(seq_len, device=x.device)
    coords = torch.cartesian_prod(t, h, w, l)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = latents.shape
    t = torch.arange(1, device=latents.device)
    h = torch.arange(height, device=latents.device)
    w = torch.arange(width, device=latents.device)
    l = torch.arange(1, device=latents.device)
    coords = torch.cartesian_prod(t, h, w, l)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def prepare_image_ids(image_latents: list[torch.Tensor], scale: int) -> torch.Tensor:
    if not isinstance(image_latents, list):
        raise ValueError("image_latents must be a list")

    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        x = x.squeeze(0)
        _, height, width = x.shape
        x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
        image_latent_ids.append(x_ids)

    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    return image_latent_ids.unsqueeze(0)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    outputs = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        height = torch.max(h_ids) + 1
        width = torch.max(w_ids) + 1
        flat_ids = h_ids * width + w_ids

        out = torch.zeros((height * width, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(height, width, ch).permute(2, 0, 1)
        outputs.append(out)

    return torch.stack(outputs, dim=0)


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)

    vae_expected_path = snapshot_dir / "vae_expected.safetensors"
    transformer_inputs_path = snapshot_dir / "transformer_inputs.safetensors"
    expected_path = snapshot_dir / "position_expected.safetensors"

    if not vae_expected_path.exists():
        raise SystemExit(f"Missing VAE expected file: {vae_expected_path}")
    if not transformer_inputs_path.exists():
        raise SystemExit(f"Missing transformer inputs file: {transformer_inputs_path}")

    vae_expected = load_file(str(vae_expected_path))
    transformer_inputs = load_file(str(transformer_inputs_path))

    latents = vae_expected["latents"]
    encoder_hidden_states = transformer_inputs["encoder_hidden_states"]

    text_ids = prepare_text_ids(encoder_hidden_states)
    latent_ids = prepare_latent_ids(latents)
    packed_latents = pack_latents(latents)
    unpacked_latents = unpack_latents_with_ids(packed_latents, latent_ids)
    image_ids = prepare_image_ids([latents], scale=args.image_scale)

    expected = {
        "text_ids": text_ids.contiguous().cpu(),
        "latent_ids": latent_ids.contiguous().cpu(),
        "image_ids": image_ids.contiguous().cpu(),
        "packed_latents": packed_latents.contiguous().cpu(),
        "unpacked_latents": unpacked_latents.contiguous().cpu(),
    }
    save_file(expected, str(expected_path))

    print(f"Wrote position fixtures to {expected_path}")


if __name__ == "__main__":
    main()
