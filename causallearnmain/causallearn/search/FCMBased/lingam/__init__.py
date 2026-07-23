"""
The lingam module includes implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

from .direct_lingam import DirectLiNGAM

_optional_import_errors = {}

try:
    from .bootstrap import (BootstrapResult, LongitudinalBootstrapResult,
                            TimeseriesBootstrapResult)
except ImportError as exc:
    _optional_import_errors["bootstrap"] = exc

try:
    from .bottom_up_parce_lingam import BottomUpParceLiNGAM
except ImportError as exc:
    _optional_import_errors["bottom_up_parce_lingam"] = exc

try:
    from .causal_effect import CausalEffect
except ImportError as exc:
    _optional_import_errors["causal_effect"] = exc

try:
    from .ica_lingam import ICALiNGAM
except ImportError as exc:
    _optional_import_errors["ica_lingam"] = exc

try:
    from .longitudinal_lingam import LongitudinalLiNGAM
except ImportError as exc:
    _optional_import_errors["longitudinal_lingam"] = exc

try:
    from .multi_group_direct_lingam import MultiGroupDirectLiNGAM
except ImportError as exc:
    _optional_import_errors["multi_group_direct_lingam"] = exc

try:
    from .rcd import RCD
except ImportError as exc:
    _optional_import_errors["rcd"] = exc

try:
    from .var_lingam import VARLiNGAM
except ImportError as exc:
    _optional_import_errors["var_lingam"] = exc

try:
    from .varma_lingam import VARMALiNGAM
except ImportError as exc:
    _optional_import_errors["varma_lingam"] = exc

__all__ = [
    name
    for name in [
        'ICALiNGAM', 'DirectLiNGAM', 'BootstrapResult', 'MultiGroupDirectLiNGAM',
        'CausalEffect', 'VARLiNGAM', 'VARMALiNGAM', 'LongitudinalLiNGAM',
        'LongitudinalBootstrapResult', 'BottomUpParceLiNGAM', 'RCD',
        'TimeseriesBootstrapResult',
    ]
    if name in globals()
]

__version__ = '1.5.4'
