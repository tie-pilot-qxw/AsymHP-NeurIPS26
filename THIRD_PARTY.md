# Third-Party Code and Attribution

AsymHP (`lb/`, `wan_t2v_sp_inference.py`) builds on the following projects.
Each component keeps its original license and copyright.

## Sparse VideoGen and Sparse VideoGen2

This repository is a fork of
[svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen),
released under the Apache License 2.0. The `svg/` package, the single-GPU
drivers (`*_inference.py`), `scripts/`, and `examples/` come from it:

- **SVG**: *Sparse VideoGen: Accelerating Video Diffusion Transformers with
  Spatial-Temporal Sparsity* (ICML 2025).
- **SVG2**: *Sparse VideoGen2: Accelerate Video Generation with Sparse
  Attention via Semantic-Aware Permutation* (NeurIPS 2025).

We use the SVG2 semantic-aware-permutation backend as the local sparse method.
Our changes to upstream files add Q/K/V and density export, per-head k-means
seeding that is independent of head placement, and more portable kernel
build settings.

## Model implementations

`svg/models/` contains adapters for open-source video diffusion models, each
under its own license:

- **Wan2.1** (Apache-2.0): `svg/models/wan/`, `svg/models/wan_orig/`.
- **HunyuanVideo** (Tencent Hunyuan Community License): `svg/models/hyvideo/`,
  `svg/models/hyvideo_orig/`.
- **CogVideoX** (Hugging Face Diffusers / THUDM): `svg/models/cog/`.
- **Cosmos** (NVIDIA): `svg/models/cosmos/`.

Model weights are downloaded from their official sources and are not included.

## Kernels and libraries

Git submodules under `svg/kernels/3rdparty/`:

- **FlashInfer** (Apache-2.0).
- **CUTLASS** (BSD-3-Clause).
- **pybind11** (BSD-3-Clause).

`lb/patches/sparge_per_head_density.patch` modifies
[SpargeAttn](https://github.com/thu-ml/SpargeAttn) (Apache-2.0) to return
per-head valid-block counts; SpargeAttn itself is not vendored.

## AsymHP

`lb/` and `wan_t2v_sp_inference.py` are original to this work and are released
under the Apache License 2.0.
