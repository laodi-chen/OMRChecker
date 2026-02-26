"""
Processor package imports.

Explicit imports help PyInstaller discover processor modules when using
pkgutil.walk_packages in src.processors.manager.
"""

# Import processor modules so they are bundled in frozen builds.
from . import CropOnMarkers  # noqa: F401
from . import CropPage  # noqa: F401
from . import FeatureBasedAlignment  # noqa: F401
from . import builtins  # noqa: F401
