# FFT Boundary Sensitivity

Minimal cross-boundary evaluation for the study *Observation-Boundary Sensitivity in FFT-Based Image Restoration: Diagnosis and Mixed-Observation Fine-Tuning*.

## Scope and status

`evaluate.py` evaluates a supplied ConverseUSRNet checkpoint on paired circular and true-context synthetic Kodak observations. It reports per-image input/output PSNR and SSIM for Gaussian and motion blur. This is an evaluation-only reproduction; it does **not** train the mixed-observation model. The two fine-tuned checkpoints and the training code are not included, so the reported mixed-fine-tuning improvements are not independently reproducible from this repository alone. No dataset, model weight, or upstream source is distributed here. This repository is a research draft, not an official ConverseNet release.

## Prerequisites

- Obtain the [ConverseNet source](https://github.com/cszn/ConverseNet) at revision `368aa39cebaa075b04926264c7494d969e8ee835` and its released `converse_usrnet.pth` by the authors' distribution route; check their terms before use. Released checkpoint SHA-256: `729529b8bc8cbda152d6b60734b01032b49c20fcb8b6da2fb33141d9c3378d82`.
- Obtain the 24 original Kodak images `kodim01.png` through `kodim24.png` independently; do not include them in this repository.
- Python with PyTorch, NumPy, OpenCV, SciPy and the dependencies required by upstream ConverseNet. The original environment used Python 3.12.10, PyTorch 2.13.0+cu130, NumPy 2.4.4, opencv-python-headless 4.12.0.88, SciPy 1.17.1 and scikit-image 0.26.0. These versions are reported for traceability, not claimed as a newly tested clean installation.

## Run

```bash
python evaluate.py --upstream /path/to/ConverseNet --checkpoint /path/to/converse_usrnet.pth --images /path/to/Kodak24 --device cuda --output results/released.csv
```

Use `--center 128 --limit 1 --device cpu` for a small smoke test; all 24 input filenames must still be present so the seeded image ordering remains consistent. Full-resolution inference has substantial GPU memory and runtime requirements. Paths are arguments; this script neither downloads datasets nor modifies source inputs. The optional checkpoint argument can accept a compatible fine-tuned state dict, but none is supplied here.

Each observation is generated from the retained image with a three-pixel surrounding support, sigma 0.01 Gaussian noise using per-image seed `12345 + index`, fixed 7×7 Gaussian or motion kernel, and no test-time tiling. The official metric functions operate on RGB predictions clipped to [0,1], scaled to 255, and rounded to uint8, with no PSNR border shave. Consult the paper protocol for the complete design and limitations.

Kodak24 informed the study design; the fine-tuning results use one seed and synthetic degradations. True-context observations are included in mixed training, so those results do not establish OOD generalization to that condition.

## License and attribution

The upstream repository is separate and its API did not report a license when this reproduction was prepared. Users must check upstream terms independently. A license for the material in this repository has not yet been selected; all rights are reserved until one is supplied.

## Verification performed

On the original local environment, a CPU smoke test on `kodim01` with `--center 128` covered both kernels and both observation boundaries. Input PSNR matched the saved reference exactly; output PSNR differed from the previously saved CUDA results by at most 0.0003 dB. This is a four-condition smoke test, not a full-dataset or clean-environment reproduction.
