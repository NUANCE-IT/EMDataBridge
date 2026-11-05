#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standardize electron-microscopy files to one metadata schema.

Files are read with RosettaSciIO (see :mod:`EMDataBridge.formats`) and their metadata
is mapped field by field. A field the file doesn't record is ``None``: an unknown
voltage is not 0 kV. An optional language-model step (``llm_backend="mock"`` or
``"llama"``) can propose values for vendor fields the explicit mapping doesn't
cover; it is off by default, and anything it adds is recorded under
``_llm_provenance``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from . import formats


class EMDataStandardizer:
    """Reads a supported file and returns its data with metadata in the standard schema."""

    # Standard schema for electron microscopy metadata. Units: voltage kV, camera_length mm,
    # exposure_time s, tilt_angle degrees, defocus µm, pixel_size nm.
    STANDARD_SCHEMA = {
        "microscope": {"name": str, "voltage": float, "magnification": float,
                       "camera_length": float, "spot_size": float, "type": str},
        "sample": {"name": str, "description": str, "preparation": str},
        "acquisition": {"date": str, "operator": str, "exposure_time": float,
                        "tilt_angle": float, "defocus": float},
        "image": {"width": int, "height": int, "bit_depth": int, "pixel_size": float},
        "processing": {"software": str, "operations": list},
    }

    def __init__(self, llm_model_name: str = "bert-base-uncased", llm_backend: str = "none",
                 llm_model_path: Optional[str] = None):
        # extension -> handler(path) returning (data, metadata) or (data, metadata, signal)
        self.supported_formats = {ext: self._read for ext in formats.READABLE}
        self.llm_backend = llm_backend or "none"
        self.llm_model_path = llm_model_path or os.environ.get("LLAMA_MODEL_PATH")
        self.llm_adapter = None
        self.llm_model_name = llm_model_name          # kept for backward compatibility

    # ------------------------------------------------------------------ reading
    def _read(self, path: Path):
        fmt = formats.format_of(path)
        signals = formats.read(path)
        sig = signals[0]
        meta = formats.to_schema(sig, path, reader=fmt.reader or "")
        if len(signals) > 1:
            meta["_source"]["signals_in_file"] = len(signals)
        return sig["data"], meta, sig

    def standardize(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Standardize one file. The first signal is returned; ``signals_in_file`` says if there are more.

        Raises:
            ValueError: the extension isn't supported.
            NotImplementedError: the format is known but has no reader yet.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            fmt = formats.BY_EXT.get(suffix)
            if fmt is not None:
                raise NotImplementedError(f"{fmt.name} ({suffix}) is not supported yet: {fmt.note}")
            raise ValueError(f"Unsupported file format: {suffix}")

        out = self.supported_formats[suffix](file_path)
        data, metadata = out[0], out[1]
        signal = out[2] if len(out) > 2 else None

        prov = self._llm_map(metadata)
        standardized = self._validate_and_standardize_metadata(metadata)
        if prov:
            standardized.setdefault("_llm_provenance", {}).update(prov)

        result = {"data": data, "metadata": standardized, "original_format": suffix[1:]}
        if signal is not None:
            result["signal"] = signal
        return result

    # ------------------------------------------------------------------ optional language model
    def _llm_map(self, metadata: Dict[str, Any]) -> Optional[dict]:
        backend = self.llm_backend.lower()
        if backend == "none":
            return None
        prefer = {"mock": "mock", "llama": "llama", "auto": "llama" if self.llm_model_path else None}.get(backend)
        if prefer is None:
            return None
        try:
            from .llm_adapter import get_llm_adapter
            if self.llm_adapter is None:
                self.llm_adapter = (get_llm_adapter(prefer="llama", model_path=self.llm_model_path)
                                    if prefer == "llama" else get_llm_adapter(prefer=prefer))
            mapping = self.llm_adapter.map_metadata(metadata)
        except Exception as e:                 # a model problem never breaks reading a file
            print(f"Warning: LLM mapping skipped: {e}")
            return None
        if not isinstance(mapping, dict):
            return None
        for cat, fields in (mapping.get("canonical") or {}).items():
            if not isinstance(fields, dict):
                continue
            target = metadata.get(cat)
            if not isinstance(target, dict):
                target = metadata[cat] = {}
            for k, v in fields.items():       # the model only fills gaps: values read from the file win
                if target.get(k) is None:
                    target[k] = v
        return mapping.get("provenance")

    # ------------------------------------------------------------------ schema
    def _validate_and_standardize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Every schema field present, each value of its type or None. Extra sections pass through."""
        out: Dict[str, Any] = {}
        for cat, fields in self.STANDARD_SCHEMA.items():
            src = metadata.get(cat) if isinstance(metadata.get(cat), dict) else {}
            out[cat] = {}
            for name, typ in fields.items():
                v = src.get(name)
                if v is not None and not isinstance(v, typ):
                    try:
                        v = typ(v)
                    except (TypeError, ValueError):
                        v = None
                if typ is str and v == "":
                    v = None
                out[cat][name] = v
        for k, v in metadata.items():
            if k not in out and k in ("signal", "_source", "_llm_provenance"):
                out[k] = v
        return out

    def save_standardized(self, standardized: Dict[str, Any], output_file: Union[str, Path],
                          format: str = "json") -> None:
        """Write the standardized metadata as JSON or YAML."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.loads(json.dumps(standardized["metadata"], default=str))
        if format.lower() == "json":
            output_file.write_text(json.dumps(metadata, indent=2))
        elif format.lower() == "yaml":
            output_file.write_text(yaml.safe_dump(metadata, default_flow_style=False, sort_keys=False, allow_unicode=True))
        else:
            raise ValueError(f"Unsupported output format: {format}")

    @staticmethod
    def list_supported_formats() -> List[Dict[str, Any]]:
        return [f.public() for f in formats.FORMATS]
