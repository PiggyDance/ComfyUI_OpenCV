"""Top-level package for ComfyUI_OpenCV."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """ComfyUI_OpenCV"""
__email__ = "narutow.cn@gmail.com"
__version__ = "0.0.1"

from .src.ComfyUI_OpenCV.nodes import NODE_CLASS_MAPPINGS
from .src.ComfyUI_OpenCV.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
