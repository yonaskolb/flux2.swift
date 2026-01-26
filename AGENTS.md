# AGENTS.md

This repository ports the FLUX.2 diffusion models to Swift with mlx-swift.

## Project goals
- Port `FLUX.2-klein-4B`, `FLUX.2-klein-9B`, and `FLUX.2-dev` to Swift with mlx-swift.
- Match the Python diffusers behavior layer-by-layer with numeric checks.
- Support quantized weights loading and inference where applicable.

## Build and test rules
- Use Swift Package Manager for all code and tests.
- Always build and test with `xcodebuild` (not `swift test`).
- Preferred pattern:
  - Build: `xcodebuild -scheme <PackageName> -destination "platform=macOS" build`
  - Test: `xcodebuild -scheme <PackageName> -destination "platform=macOS" test`

## Planning files
- Store planning artifacts in `mem/` (e.g., `mem/task_plan.md`, `mem/notes.md`, `mem/project_plan.md`).

## Porting plan (high level)
1. **Scaffold the Swift package**
   - Define package layout, targets, and test targets in `Package.swift`.
   - Add minimal build/test scripts or documentation for `xcodebuild`.
2. **Model architecture parity**
   - Mirror module structure from `diffusers` (naming and shapes).
3. **Core layers and ops**
   - Implement core layers in Swift with mlx-swift (attention, MLP, normalization).
   - Validate each layer against Python outputs using fixed inputs.
4. **Tokenizer + VAE + UNet/Transformer blocks**
   - Port components in dependency order.
   - Add per-block numerical checks (golden outputs).
5. **End-to-end pipeline**
   - Assemble scheduler, denoising loop, and sampling logic.
   - Compare final outputs to Python reference on known seeds.
6. **Quantization**
   - Reuse patterns from `zimage.swift` for quantized weights.
   - Provide both FP and quantized loading paths with tests.
7. **Scale-up**
   - Start with `FLUX.2-klein-4B`, then `FLUX.2-klein-9B`, then `FLUX.2-dev`.
   - Reuse shared components and keep config-driven shape definitions.

## Validation approach
- Create small deterministic test vectors in Python and save to fixtures.
- Add Swift tests that load fixtures and compare numerically (tolerances noted).
- Track CPU/GPU transfer points to avoid unintended host-device copies.

## Contribution guidelines
- Keep APIs Swift-first but stay faithful to reference shapes and semantics.
- Prefer small, testable units and add tests with each layer.
- Document any intentional deviations from the reference implementation.
