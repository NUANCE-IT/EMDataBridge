#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMDataBridge: A bridge between electron microscopy data formats

This package provides tools for standardizing, translating, and discovering
electron microscopy data formats and their metadata, and the researcher
identity (ORCID, signing keys, trust) shared by the NUANCE tools.

The format tools load lazily, so ``from EMDataBridge import identity`` stays
light (standard library and ``cryptography`` only).
"""

__version__ = "0.1.0"
__author__ = "Roberto dos Reis"
__email__ = "robertomsreis@gmail.com"
__url__ = "https://github.com/NUANCE-IT/EMDataBridge"

from . import identity  # noqa: E402,F401

_LAZY = {
    "EMDataStandardizer": ".standardizer",
    "EMFormatTranslator": ".translator",
    "EMDataDiscovery": ".discovery",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["identity", *_LAZY]
