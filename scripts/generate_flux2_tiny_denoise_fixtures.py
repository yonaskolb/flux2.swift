#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2 denoise step fixtures."
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_tiny",
        help="Output directory for fixtures.",
    )
    parser.add_argument(
        "--transformer-dir",
        default="fixtures/flux2_tiny/transformer",
        help="Path to tiny transformer snapshot.",
    )
    parser.add_argument(
        "--inputs",
        default="fixtures/flux2_tiny/transformer_inputs.safetensors",
        help="Inputs safetensors containing hidden_states and ids.",
    )
    parser.add_argument(
        "--diffusers-path",
        default="temp/diffusers/src",
        help="Path to diffusers src/ for the reference implementation.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=4,
        help="Number of inference steps for scheduler.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for transformer inputs/outputs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for running the reference model.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def main() -> None:
    args = parse_args()
    diffusers_src = Path(args.diffusers_path)
    if diffusers_src.exists():
        sys.path.insert(0, str(diffusers_src))
    else:
        raise SystemExit(f"diffusers path not found: {diffusers_src}")

    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    transformer_dir = Path(args.transformer_dir)
    with (transformer_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    transformer = Flux2Transformer2DModel(**config)
    weights = load_file(str(transformer_dir / "model.safetensors"))
    transformer.load_state_dict(weights)
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()

    scheduler_config_path = Path(args.out_dir) / "scheduler" / "config.json"
    with scheduler_config_path.open("r", encoding="utf-8") as f:
        scheduler_config = json.load(f)
    scheduler_config = {
        key: value for key, value in scheduler_config.items() if not key.startswith("_")
    }
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    inputs = load_file(args.inputs)
    latents = inputs["hidden_states"].to(device=device, dtype=dtype)
    encoder_hidden_states = inputs["encoder_hidden_states"].to(device=device, dtype=dtype)
    img_ids = inputs["img_ids"].to(device=device)
    txt_ids = inputs["txt_ids"].to(device=device)

    timestep = scheduler.timesteps[:1].clone().to(device=device, dtype=dtype)

    with torch.no_grad():
        noise_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )[0]
        prev_latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_out = out_dir / "denoise_inputs.safetensors"
    expected_out = out_dir / "denoise_expected.safetensors"

    save_file(
        {
            "latents": latents.contiguous().cpu(),
            "encoder_hidden_states": encoder_hidden_states.contiguous().cpu(),
            "timestep": timestep.contiguous().cpu(),
            "img_ids": img_ids.contiguous().cpu(),
            "txt_ids": txt_ids.contiguous().cpu(),
        },
        str(inputs_out),
    )

    save_file(
        {
            "noise_pred": noise_pred.contiguous().cpu(),
            "prev_latents": prev_latents.contiguous().cpu(),
        },
        str(expected_out),
    )

    print(f"Wrote inputs to {inputs_out}")
    print(f"Wrote expected outputs to {expected_out}")


if __name__ == "__main__":
    main()
