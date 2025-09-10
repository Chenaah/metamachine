# MetaMachine

[![PyPI version](https://badge.fury.io/py/metamachine.svg)](https://badge.fury.io/py/metamachine)
[![Python versions](https://img.shields.io/pypi/pyversions/metamachine.svg)](https://pypi.org/project/metamachine/)
[![License](https://img.shields.io/pypi/l/metamachine.svg)](https://github.com/chenaah/metamachine/blob/main/LICENSE)
[![CI](https://github.com/chenaah/metamachine/workflows/CI/badge.svg)](https://github.com/chenaah/metamachine/actions)
[![codecov](https://codecov.io/gh/chenaah/metamachine/branch/main/graph/badge.svg)](https://codecov.io/gh/chenaah/metamachine)

A simulation framework for modular robots, designed to accelerate research in robot learning and evolutionary robotics.

## Overview

MetaMachine is a simulation framework for modular robotic systems. Currently featuring a **MuJoCo-based modular legs** implementation based on my [recent work](https://arxiv.org/abs/2505.00784) on reconfigurable legged metamachines, this framework will expand to support diverse modular robot architectures using various simulation backends throughout my PhD research.

**Vision**: MetaMachine aims to become a cornerstone in the next-generation ecosystem for fast prototyping robot learning research systems, serving as both a robot learning and evolutionary robotics benchmark and comprehensive development platform.

## Current Implementation

### 🦿 Modular Legs System
Based on "[Reconfigurable legged metamachines that run on autonomous modular legs](https://arxiv.org/abs/2505.00784)" (arXiv:2505.00784):
- **Autonomous modular legs**: Single-degree-of-freedom jointed links that learn complex dynamic behaviors
- **Reconfigurable metamachines**: Freely attachable modules forming meter-scale legged robots
- **Dynamic locomotion**: Non-quasistatic movement through unstructured environments

## Key Features

- **🎯 Research-focused**: Designed for robot learning and evolutionary robotics research
- **🔧 Modular architecture**: Extensible framework for diverse robot systems
- **⚡ Physics simulation**: High-fidelity simulation (current: MuJoCo)
- **🔬 Gymnasium-compatible**: Standard RL environment interface
- **📊 Rich visualization**: Real-time rendering and video recording

## Quick Start

### Installation

#### From PyPI (Recommended)

```bash
# Install the latest stable release
pip install metamachine

# Or install with optional dependencies
pip install metamachine[dev]  # Development tools
pip install metamachine[docs] # Documentation tools
```

#### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/chenaah/metamachine
cd metamachine

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

#### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Dependencies**: NumPy, Gymnasium, MuJoCo, OmegaConf

#### Verify Installation

```bash
python -c "import metamachine; print('MetaMachine installed successfully!')"
```

#### Troubleshooting

**MuJoCo Installation Issues:**
```bash
# On Linux, you may need additional system dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# On macOS with Apple Silicon
pip install mujoco --no-binary mujoco
```

**Import Errors:**
```bash
# Ensure you have the latest pip
pip install --upgrade pip

# Clean install if you encounter issues
pip uninstall metamachine
pip install metamachine
```

### Basic Usage

```python
import numpy as np
from metamachine.environments.configs.config_registry import ConfigRegistry
from metamachine.environments.env_sim import MetaMachine

# Create modular legs environment
cfg = ConfigRegistry.create_from_name("basic_quadruped")
env = MetaMachine(cfg)

# Run simulation
env.reset(seed=42)
for step in range(1000):
    action = np.random.randn(env.action_space.shape[0])
    obs, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        break
```

## Development Roadmap

MetaMachine is actively being developed with plans to expand beyond modular legs:

- **🚀 Current**: Modular legs implementation (based on arXiv:2505.00784)
- **🔮 Upcoming**: Additional modular robot architectures during PhD research
- **🎯 Long-term**: Comprehensive benchmark suite for robot learning and evolutionary robotics
- **🌟 Vision**: Core component of next-gen robot learning research ecosystem

## Project Structure

```
metamachine/
├── environments/          # Core simulation environments
│   ├── components/        # Modular environment components
│   └── configs/          # Configuration system
├── robot_factory/        # Robot design and generation
│   └── modular_legs/     # Current: Modular leg system
└── utils/                # Utility functions and helpers
```

## Configuration

Simple configuration system for different robot setups:

```python
from metamachine.environments.configs.config_registry import ConfigRegistry

# Load predefined configuration
cfg = ConfigRegistry.create_from_name("basic_quadruped")

# Customize as needed
cfg.simulation.timestep = 0.01
cfg.control.action_scale = 2.0
```

## Contributing

This is an active research project. Contributions, feedback, and collaborations are welcome!

### Development Setup

1. Fork and clone the repository
2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```
4. Create a feature branch and make your changes
5. Run tests and ensure code quality:
   ```bash
   pytest
   black metamachine/ tests/
   ruff check metamachine/ tests/
   ```
6. Submit a pull request

### Release Status

- 🔄 **Active Development**: This package is under active development
- 📦 **PyPI Available**: Stable releases are published to PyPI
- 🧪 **Research Focus**: Designed primarily for research applications
- 📈 **Expanding**: More modular robot architectures coming soon

## Citation

If you use MetaMachine in your research, please cite:

```bibtex
@software{metamachine2025,
  title={MetaMachine: A Simulation Framework for Modular Robots},
  author={Chen Yu},
  year={2025},
  url={https://github.com/chenaah/metamachine},
  note={Available on PyPI: \url{https://pypi.org/project/metamachine/}}
}
```

For the modular legs implementation, please also cite:

```bibtex
@article{yu2025reconfigurable,
  title={Reconfigurable legged metamachines that run on autonomous modular legs},
  author={Yu, Chen and Matthews, David and Wang, Jingxian and Gu, Jing and Blackiston, Douglas and Rubenstein, Michael and Kriegman, Sam},
  journal={arXiv preprint arXiv:2505.00784},
  year={2025}
}
```

## License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.
