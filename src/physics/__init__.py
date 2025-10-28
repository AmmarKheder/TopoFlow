"""
Physics-based attention modules for atmospheric pollution modeling.

This package implements physically-grounded attention mechanisms including:
- Richardson Number computation for atmospheric stability
- Elevation-based topographic barriers
- Wind-aware spatial coordinate tracking
"""

from .richardson_mask import RichardsonPhysicsMask
from .spatial_coordinates import SpatialCoordinateTracker
from .physics_utils import compute_patch_elevations, compute_wind_patches

__all__ = [
    'RichardsonPhysicsMask',
    'SpatialCoordinateTracker',
    'compute_patch_elevations',
    'compute_wind_patches',
]
