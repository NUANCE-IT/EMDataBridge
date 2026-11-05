#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate electron-microscopy files between formats.

The data and its calibrated axes are written exactly as read: no rescaling and no
change of number type. If the target format can't hold the data as it is (an MRC
file can't store float64, an MSA file holds one spectrum), translation stops with
a clear error instead of converting silently.

Most formats can't carry all of the source metadata. Unless the target keeps
everything (HyperSpy, HDF5), the standardized metadata is also written beside the
output as ``<output>.emdb.json``, so nothing is lost without a trace.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import formats
from .standardizer import EMDataStandardizer


class EMFormatTranslator:
    """Converts a supported file to another format, keeping pixels and axes untouched."""

    def __init__(self, llm_model_name: str = "bert-base-uncased", llm_backend: str = "none",
                 llm_model_path: Optional[str] = None):
        self.standardizer = EMDataStandardizer(llm_model_name=llm_model_name, llm_backend=llm_backend,
                                               llm_model_path=llm_model_path)
        self.output_formats = dict(formats.WRITABLE)          # extension (no dot) -> Format

    def translate(self, input_file: Union[str, Path], output_file: Union[str, Path],
                  output_format: Optional[str] = None, sidecar: bool = True) -> Dict[str, Any]:
        """Translate one file. Returns what was written.

        Raises:
            ValueError: the input or output format isn't supported, or the data doesn't fit the target.
        """
        input_file, output_file = Path(input_file), Path(output_file)
        ext = (output_format or output_file.suffix).lower().lstrip(".")
        if ext not in self.output_formats:
            formats.write({"data": [0]}, output_file, ext)    # raises the explanatory FormatError
        std = self.standardizer.standardize(input_file)
        signal = std.get("signal") or {"data": std["data"]}
        fmt = formats.write(signal, output_file, ext)
        out = {"input": str(input_file), "output": str(output_file), "format": fmt.name,
               "kept_in_file": fmt.metadata or "data only"}
        if sidecar and fmt.metadata != "everything":
            side = output_file.with_name(output_file.name + ".emdb.json")
            meta = {**std["metadata"], "_translation": {"from": input_file.name, "to": output_file.name,
                                                         "format": fmt.name, "kept_in_file": out["kept_in_file"]}}
            side.write_text(json.dumps(meta, indent=2, default=str))
            out["sidecar"] = str(side)
        print(f"Translated {input_file} to {output_file} ({fmt.name})")
        return out

    def batch_translate(self, input_dir: Union[str, Path], output_dir: Union[str, Path],
                        output_format: str, file_pattern: str = "*") -> List[Dict[str, Any]]:
        """Translate every readable file in a directory. Failures are reported and skipped."""
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_format = output_format.lower().lstrip(".")
        if output_format not in self.output_formats:
            formats.write({"data": [0]}, output_dir / f"x.{output_format}", output_format)
        files = [f for f in sorted(input_dir.glob(file_pattern)) if f.suffix.lower() in self.standardizer.supported_formats]
        if not files:
            print(f"No supported files found in {input_dir} matching pattern {file_pattern}")
            return []
        print(f"Found {len(files)} files to translate")
        done = []
        for i, f in enumerate(files, 1):
            try:
                done.append(self.translate(f, output_dir / f"{f.stem}.{output_format}", output_format))
            except Exception as e:
                print(f"Error translating {f.name}: {e}")
        print(f"Translated {len(done)} of {len(files)}")
        return done

    @staticmethod
    def list_supported_formats() -> List[Dict[str, Any]]:
        """Every known format with its status: read, read+write or planned."""
        return [f.public() for f in formats.FORMATS]
