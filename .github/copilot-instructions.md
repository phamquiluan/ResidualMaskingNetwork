# ResidualMaskingNetwork Copilot Instructions

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Installation
- Install dependencies manually (bypass installation issues):
  - `pip install numpy opencv-python torch torchvision requests pytorchcv tqdm`  -- takes 2-3 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
  - `pip install imgaug matplotlib scipy scikit-image` -- required for training scripts.
- Build from source installation has issues with version parsing in setup.py (reads wrong line for version). Use manual dependency installation instead.
- The rmn package can be imported directly by adding the repository root to Python path: `import sys; sys.path.insert(0, '.'); from rmn import RMN`

### Testing Functionality
- Test basic inference functionality:
  - Download test image: `curl "https://user-images.githubusercontent.com/24642166/117108145-f7e6a480-adac-11eb-9786-a3e50a3cdea6.jpg" -o "test_image.png"`
  - Run inference: `python -c "import sys; sys.path.insert(0, '.'); import cv2; from rmn import RMN; m = RMN(); results = m.detect_emotion_for_single_frame(cv2.imread('test_image.png')); print(results)"`
- On first import, rmn automatically downloads required model files (~550MB total). This is normal and expected.

### Training Scripts
- Main training script: `python main_fer2013.py` (requires FER2013 dataset)
- Training requires additional dependencies: `imgaug`, `matplotlib`, `scipy`, `scikit-image`
- Note: Current imgaug version has compatibility issues with numpy >= 2.0. Use numpy < 2.0 if needed for training.
- Configuration files are in `configs/` directory. Default uses `configs/fer2013_config.json`
- Training creates checkpoints in `saved/checkpoints/` directory
- Training logs are saved to `log/` directory with TensorBoard format

## Validation
- ALWAYS test basic inference functionality after making changes to ensure the package still works
- Always run linting tools before submitting code:
  - `black --check .` -- code formatting check, takes ~2 seconds
  - `flake8 .` -- linting check, takes ~1 second  
  - `isort --check .` -- import sorting check, takes ~1 second
- The repository has known linting issues in some files. Only fix linting issues related to your changes.
- CI/CD validation: The GitHub Actions workflow tests basic package installation and inference

## Common Commands and Expected Times
- Installing dependencies: 2-3 minutes (torch and related packages are large)
- Black formatting check: ~2 seconds
- Flake8 linting: ~1 second
- Isort import sorting: ~1 second
- Basic inference test: ~5-10 seconds (first run includes model download)
- Subsequent inference tests: ~1-2 seconds

## Important Repository Structure
### Core directories:
- `rmn/` - Main inference package (production code)
- `models/` - Model architecture definitions
- `trainers/` - Training logic and implementations
- `utils/` - Utility functions for datasets, metrics, etc.
- `configs/` - Training configuration files
- `script/` - Helper scripts for evaluation and results generation
- `legacy/` - Legacy training scripts for different datasets

### Key files:
- `main_fer2013.py` - Main training script for FER2013 dataset
- `setup.py` - Package setup (has version parsing issues)
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `.flake8` - Flake8 linting configuration
- `pyproject.toml` - Black formatting configuration

## Common Issues and Workarounds
- **Installation Issue**: pip install -e . fails due to version parsing in setup.py. Use manual dependency installation instead.
- **Training Import Error**: Missing imgaug dependency. Install with `pip install imgaug`
- **NumPy Compatibility**: imgaug has issues with numpy >= 2.0. Use numpy < 2.0 for training if needed.
- **Model Download**: First rmn import downloads ~550MB of models. This is expected behavior.
- **Network Timeouts**: pip installations may timeout. Use `--timeout 120` or higher for large packages.

## Linting Configuration
- **Black**: Line length 88, configured in `pyproject.toml`
- **Flake8**: Ignores E203, E266, E501, W503, F403, F401. Max line length 100. Configuration in `.flake8`
- **Isort**: Black-compatible profile, line length 88. Configuration in `.isort.cfg`
- **Pre-commit**: Configured with black, flake8, isort, and other hooks in `.pre-commit-config.yaml`

## CI/CD Workflow
- GitHub Actions workflow in `.github/workflows/python-package.yml`
- Tests Python versions 3.8 through 3.12
- Installs package with `pip install -e .`
- Validates basic import: `python -c "from rmn import RMN"`
- Tests inference on sample image
- Takes approximately 2-3 minutes to complete

## Manual Validation Requirements
After making changes:
1. Test basic package import: `python -c "import sys; sys.path.insert(0, '.'); from rmn import RMN"`
2. Test inference functionality with the commands listed above
3. Run linting tools: `black --check .`, `flake8 .`, `isort --check .`
4. If modifying training code, test that main_fer2013.py can start (will fail without dataset but should not have import errors)

Always prioritize functionality over perfect linting scores. The repository has existing linting issues that should not block functional changes.