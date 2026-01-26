#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny FlowMatch Euler scheduler fixtures."
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_tiny",
        help="Output directory for scheduler fixtures.",
    )
    parser.add_argument(
        "--vae-expected",
        default="fixtures/flux2_tiny/vae_expected.safetensors",
        help="Path to VAE expected safetensors containing latents.",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    vae_expected = load_file(args.vae_expected)
    latents = vae_expected["latents"].to(dtype=torch.float32)

    diffusers_src = Path(args.diffusers_path)
    if diffusers_src.exists():
        sys.path.insert(0, str(diffusers_src))
    else:
        raise SystemExit(f"diffusers path not found: {diffusers_src}")

    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(args.num_inference_steps, device="cpu")

    noise = torch.randn_like(latents)
    model_output = torch.randn_like(latents)

    timestep = scheduler.timesteps[:1].clone()
    scaled_sample = scheduler.scale_noise(latents, timestep, noise)
    prev_sample = scheduler.step(model_output, scheduler.timesteps[0], latents).prev_sample

    config = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.0.0",
        "num_train_timesteps": scheduler.config.num_train_timesteps,
        "shift": scheduler.config.shift,
        "use_dynamic_shifting": scheduler.config.use_dynamic_shifting,
        "base_shift": scheduler.config.base_shift,
        "max_shift": scheduler.config.max_shift,
        "base_image_seq_len": scheduler.config.base_image_seq_len,
        "max_image_seq_len": scheduler.config.max_image_seq_len,
        "invert_sigmas": scheduler.config.invert_sigmas,
        "shift_terminal": scheduler.config.shift_terminal,
        "use_karras_sigmas": scheduler.config.use_karras_sigmas,
        "use_exponential_sigmas": scheduler.config.use_exponential_sigmas,
        "use_beta_sigmas": scheduler.config.use_beta_sigmas,
        "time_shift_type": scheduler.config.time_shift_type,
        "stochastic_sampling": scheduler.config.stochastic_sampling,
    }

    out_dir = Path(args.out_dir)
    scheduler_dir = out_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)

    with (scheduler_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    expected = {
        "timesteps": scheduler.timesteps.contiguous().cpu(),
        "sigmas": scheduler.sigmas.contiguous().cpu(),
        "timestep": timestep.contiguous().cpu(),
        "sample": latents.contiguous().cpu(),
        "noise": noise.contiguous().cpu(),
        "model_output": model_output.contiguous().cpu(),
        "scaled_sample": scaled_sample.contiguous().cpu(),
        "prev_sample": prev_sample.contiguous().cpu(),
    }
    expected_path = out_dir / "scheduler_expected.safetensors"
    save_file(expected, str(expected_path))

    print(f"Wrote scheduler fixtures to {expected_path}")


if __name__ == "__main__":
    main()
