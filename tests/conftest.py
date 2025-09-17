"""
Pytest configuration for MetaMachine tests.

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import pytest


def pytest_configure(config):
    """Configure pytest for headless environments."""
    # Disable all rendering for tests
    os.environ["MUJOCO_GL"] = "disable"

    # Backup: try osmesa if disable doesn't work
    if "MUJOCO_GL" not in os.environ or os.environ["MUJOCO_GL"] == "":
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    # Disable GUI-related features
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Prevent any OpenGL context creation
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["GALLIUM_DRIVER"] = "softpipe"


@pytest.fixture(scope="session", autouse=True)
def setup_headless_environment():
    """Set up headless environment for all tests."""
    # Ensure no rendering is attempted
    original_gl = os.environ.get("MUJOCO_GL")

    # Force no rendering
    os.environ["MUJOCO_GL"] = "disable"

    yield

    # Restore original value if it existed
    if original_gl is not None:
        os.environ["MUJOCO_GL"] = original_gl
    elif "MUJOCO_GL" in os.environ:
        del os.environ["MUJOCO_GL"]


@pytest.fixture
def no_render_config():
    """Provide configuration that completely disables rendering."""
    return {
        "render_mode": None,
        "enable_viewer": False,
        "record_video": False,
        "headless": True,
    }
