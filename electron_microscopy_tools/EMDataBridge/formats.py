"""Reading and writing electron-microscopy files, and mapping their metadata to one schema.

Files are read with RosettaSciIO (the HyperSpy file readers, without HyperSpy), which
returns each signal as a dict: the array, its axes (name, size, scale, offset, units),
the metadata in HyperSpy's layout, and the vendor's original metadata. Writing goes
back through RosettaSciIO (or mrcfile / h5py), so pixels and calibrated axes pass
through untouched.

Metadata mapping is explicit and deterministic: every field in the schema comes from a
named place in the file, and a field the file doesn't record is ``None``, never a
made-up default. ``FORMATS`` says what each format supports today; formats with no
reader yet are listed as planned rather than claimed.

Created by Roberto dos Reis (@rmsreis), NUANCE Center, Northwestern University.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Format:
    name: str
    extensions: tuple[str, ...]
    reader: Optional[str] = None      # "rsciio.<plugin>", "h5py" or None (planned)
    writer: Optional[str] = None      # "rsciio.<plugin>", "mrcfile", "h5py" or None
    kinds: str = "any"                # what the writer accepts: "any", "image", "spectrum"
    metadata: str = ""                # what the format itself keeps when written
    note: str = ""

    @property
    def status(self) -> str:
        if self.reader and self.writer:
            return "read, write"
        return "read" if self.reader else ("write" if self.writer else "planned")

    def public(self) -> dict:
        return {"name": self.name, "extensions": list(self.extensions), "status": self.status,
                "reader": self.reader, "writer": self.writer, "note": self.note}


FORMATS: tuple[Format, ...] = (
    Format("Gatan Digital Micrograph", (".dm3", ".dm4"), "rsciio.digitalmicrograph",
           note="Read only: no open-source library writes DM files"),
    Format("Velox / Berkeley EMD", (".emd",), "rsciio.emd", "rsciio.emd", metadata="axes",
           note="Reads Thermo Fisher Velox and Berkeley (NCEM) EMD; writes Berkeley EMD"),
    Format("Thermo Fisher TIA", (".ser", ".emi"), "rsciio.tia"),
    Format("EMSA/MSA spectrum", (".msa", ".emsa", ".ems", ".mas"), "rsciio.msa", "rsciio.msa", kinds="spectrum",
           metadata="axes, beam energy, date", note="Writes single spectra (one-dimensional data)"),
    Format("MRC", (".mrc", ".mrcs", ".ali", ".rec", ".st"), "rsciio.mrc", "mrcfile", metadata="voxel size",
           note="Tomography stacks (.ali, .rec, .st) included"),
    Format("TIFF", (".tif", ".tiff"), "rsciio.tiff", "rsciio.tiff", metadata="pixel size",
           note="Reads Thermo Fisher (FEI), Zeiss, TVIPS, Olympus, JEOL SightX and Hamamatsu metadata"),
    Format("HyperSpy", (".hspy",), "rsciio.hspy", "rsciio.hspy", metadata="everything",
           note="Keeps all metadata and the vendor's original metadata"),
    Format("HDF5", (".h5", ".hdf5"), "h5py", "h5py", metadata="everything",
           note="HyperSpy files are read in full; other HDF5 files by their largest dataset"),
    Format("Bruker", (".bcf", ".spx"), "rsciio.bruker"),
    Format("JEOL", (".asw", ".pts", ".map", ".eds"), "rsciio.jeol"),
    Format("EDAX", (".spd", ".spc"), "rsciio.edax"),
    Format("NanoMegas blockfile", (".blo",), "rsciio.blockfile"),
    Format("TVIPS", (".tvips",), "rsciio.tvips"),
    Format("Quantum Detectors MIB", (".mib",), "rsciio.quantumdetector"),
    Format("EM (TOM/EMAN)", (".em",), note="Planned"),
    Format("Hitachi", (".hds",), note="Planned: needs sample files"),
    Format("Oxford Instruments AZtec", (".h5oina", ".oip", ".azw", ".azd"), note="Planned: needs sample files"),
)

BY_EXT: dict[str, Format] = {e: f for f in FORMATS for e in f.extensions}
READABLE = tuple(e for e, f in BY_EXT.items() if f.reader)
WRITABLE = {e.lstrip("."): f for e, f in BY_EXT.items() if f.writer}


class FormatError(ValueError):
    pass


def format_of(path: str | Path) -> Format:
    ext = Path(path).suffix.lower()
    if ext not in BY_EXT:
        raise FormatError(f"Unsupported file format: {ext or '(no extension)'}")
    return BY_EXT[ext]


def _plugin(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ImportError(f"{name} is needed for this format: pip install 'EMDataBridge[io]' ({e})") from e


# ------------------------------------------------------------------ signals
def complete(signal: dict) -> dict:
    """Fill in the keys RosettaSciIO's writers expect, without touching data or axes."""
    from . import __version__

    s = dict(signal)
    data = np.asarray(s["data"])
    s.setdefault("axes", [{"name": f"axis{i}", "size": n, "scale": 1.0, "offset": 0.0, "units": "", "navigate": False}
                          for i, n in enumerate(data.shape)])
    md = json.loads(json.dumps(s.get("metadata") or {}, default=str))
    md.setdefault("General", {}).setdefault("title", "")
    md.setdefault("Signal", {}).setdefault("signal_type", "")
    md["Signal"].setdefault("quantity", "")
    s["metadata"] = md
    s.setdefault("original_metadata", {})
    s.setdefault("tmp_parameters", {})
    s.setdefault("learning_results", {})
    s.setdefault("models", {})
    s["attributes"] = {"_lazy": False, **(s.get("attributes") or {})}
    s.setdefault("package_info", {"name": "EMDataBridge", "version": __version__})
    return s


def read(path: str | Path) -> list[dict]:
    """All signals in a file, each as a RosettaSciIO signal dict."""
    path = Path(path)
    fmt = format_of(path)
    if not fmt.reader:
        raise NotImplementedError(f"{fmt.name} ({path.suffix}) is not supported yet: {fmt.note}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if fmt.reader == "h5py":
        return _read_hdf5(path)
    return [complete(apply_mapping(s)) for s in _plugin(fmt.reader).file_reader(str(path), lazy=False)]


# Fields RosettaSciIO's own mappings leave out, as (vendor path, metadata path). "{scope}" is TEM or SEM.
EXTRA_MAPPING = {
    "ImageList.TagGroup0.ImageTags.Microscope Info.Name": "Acquisition_instrument.{scope}.microscope",
    "ImageList.TagGroup0.ImageTags.DataBar.Exposure Time (s)": "Acquisition_instrument.{scope}.Detector.Camera.exposure",
    "ImageList.TagGroup0.ImageTags.Microscope Info.STEM Camera Length": "Acquisition_instrument.{scope}.camera_length",
    "ImageList.TagGroup0.ImageTags.Microscope Info.Indicated Magnification": "Acquisition_instrument.{scope}.magnification",
}


def _lookup(tree: Any, path: str):
    """Follow a dotted vendor path; 'Microscope Info' and 'Microscope_Info' are the same key (as in HyperSpy)."""
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        if part in node:
            node = node[part]
            continue
        want = part.replace(" ", "_")
        node = next((v for k, v in node.items() if str(k).replace(" ", "_") == want), None)
        if node is None:
            return None
    return node


def _put(tree: dict, path: str, value) -> None:
    *parents, last = path.split(".")
    for p in parents:
        tree = tree.setdefault(p, {})
    if tree.get(last) is None:                        # what the reader already set wins, unless it is empty
        tree[last] = value


def apply_mapping(signal: dict) -> dict:
    """Copy instrument values from the vendor tags into the metadata, as HyperSpy does on load.

    RosettaSciIO returns the vendor-to-metadata table as ``signal["mapping"]``; HyperSpy normally
    applies it. Without HyperSpy we apply it here, then add the few fields in EXTRA_MAPPING.
    """
    om = signal.get("original_metadata") or {}
    md = signal.setdefault("metadata", {})
    for src, (dst, fn) in (signal.pop("mapping", None) or {}).items():
        v = _lookup(om, src)
        if v is None:
            continue
        try:
            _put(md, dst, fn(v) if fn else v)
        except Exception:                             # one odd tag never stops a file from loading
            continue
    inst = md.get("Acquisition_instrument") or {}
    scope = "SEM" if "SEM" in inst else "TEM"
    for src, dst in EXTRA_MAPPING.items():
        v = _lookup(om, src)
        if v is not None and not isinstance(v, dict):
            _put(md, dst.format(scope=scope), v)
    return signal


def _read_hdf5(path: Path) -> list[dict]:
    h5py = _plugin("h5py")
    with h5py.File(path, "r") as f:
        if str(f.attrs.get("file_format", "")).lower() == "hyperspy":
            return [complete(s) for s in _plugin("rsciio.hspy").file_reader(str(path), lazy=False)]
        if "emdatabridge" in f.attrs:                       # written by write() below
            sig = json.loads(f.attrs["emdatabridge"])
            sig["data"] = f["data"][()]
            return [complete(sig)]
        found: list = []
        f.visititems(lambda name, obj: found.append((obj.size, name)) if isinstance(obj, h5py.Dataset) and obj.ndim >= 1 else None)
        if not found:
            raise FormatError(f"No dataset in {path.name}")
        name = max(found)[1]
        data = f[name][()]
        attrs = {k: _plain(v) for k, v in f[name].attrs.items()}
    return [complete({"data": data, "metadata": {"General": {"title": name}}, "original_metadata": {name: attrs}})]


def _plain(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def write(signal: dict, path: str | Path, fmt: Optional[str] = None) -> Format:
    """Write one signal. Pixels and axes are written as they are: no rescaling, no type change."""
    path = Path(path)
    ext = (fmt or path.suffix).lower().lstrip(".")
    if ext not in WRITABLE:
        known = BY_EXT.get("." + ext)
        why = f"{known.name} can't be written: {known.note}" if known else f"Unsupported output format: {ext}"
        raise FormatError(why)
    f = WRITABLE[ext]
    s = complete(signal)
    data = np.asarray(s["data"])
    if f.kinds == "spectrum" and data.ndim != 1:
        raise FormatError(f"{f.name} holds a single spectrum; this data has shape {data.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if f.writer == "mrcfile":
        mrcfile = _plugin("mrcfile")
        if data.dtype not in (np.int8, np.uint8, np.int16, np.uint16, np.float32, np.complex64):
            raise FormatError(f"MRC can't store {data.dtype} without converting it; write HDF5 or HyperSpy instead")
        with mrcfile.new(path, overwrite=True) as m:
            m.set_data(data)
            px = pixel_size_nm(s)
            if px:
                m.voxel_size = px * 10                      # MRC stores Ångström
    elif f.writer == "h5py":
        h5py = _plugin("h5py")
        with h5py.File(path, "w") as h:
            h.create_dataset("data", data=data)
            keep = {k: s[k] for k in ("axes", "metadata", "original_metadata")}
            h.attrs["emdatabridge"] = json.dumps(keep, default=str)
    else:
        _plugin(f.writer).file_writer(str(path), s)
    return f


# ------------------------------------------------------------------ metadata
UNITS_NM = {"pm": 1e-3, "Å": 0.1, "A": 0.1, "ångström": 0.1, "nm": 1.0, "µm": 1e3, "um": 1e3, "μm": 1e3,
            "micron": 1e3, "mm": 1e6, "m": 1e9}

SCHEMA_FIELDS = {
    "microscope": ("name", "voltage", "magnification", "camera_length", "spot_size"),
    "sample": ("name", "description", "preparation"),
    "acquisition": ("date", "operator", "exposure_time", "tilt_angle", "defocus"),
    "image": ("width", "height", "bit_depth", "pixel_size"),
    "processing": ("software", "operations"),
}


def _get(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


ENERGY_UNITS = {"ev", "kev", "mev"}


def _is_energy(axis: dict) -> bool:
    return str(axis.get("units") or "").strip().lower() in ENERGY_UNITS


def signal_axes(signal: dict) -> list[dict]:
    """Signal (not navigation) axes, in array order."""
    return [a for a in signal["axes"] if not a.get("navigate", False)]


def image_axes(signal: dict) -> list[dict]:
    """The axes that make up one image, in array order (y, x).

    Readers don't always flag a spectrum image's energy axis as the signal axis (a DM
    EELS SI can come back with energy first, flagged as navigation), so when there is
    an energy axis, the image axes are the others.
    """
    axes = signal["axes"]
    if any(_is_energy(a) for a in axes):
        return [a for a in axes if not _is_energy(a)][-2:]
    return signal_axes(signal)


def pixel_size_nm(signal: dict) -> Optional[float]:
    ax = image_axes(signal)
    if not ax:
        return None
    k = UNITS_NM.get(str(ax[-1].get("units") or "").strip())      # 1/nm (diffraction) or no units: not a length
    scale = ax[-1].get("scale")
    return float(scale) * k if k and scale is not None else None


def kind(signal: dict) -> str:
    axes = signal["axes"]
    energy = sum(_is_energy(a) for a in axes)
    if energy == 1:
        return {1: "spectrum", 2: "spectrum line", 3: "spectrum image"}.get(len(axes), "spectrum series")
    sig = len(signal_axes(signal))
    nav = len(axes) - sig
    base = {0: "point", 1: "spectrum", 2: "image"}.get(sig, f"{sig}d")
    if not nav:
        return base
    return {"spectrum": "spectrum image" if nav == 2 else "spectrum line",
            "image": "image stack" if nav == 1 else "4D dataset"}.get(base, f"{base} series")


def to_schema(signal: dict, path: Optional[Path] = None, reader: str = "") -> dict:
    """Map one signal's metadata to the EMDataBridge schema. Unknown is None."""
    md = signal.get("metadata") or {}
    inst = _get(md, "Acquisition_instrument") or {}
    scope_key = "TEM" if "TEM" in inst else ("SEM" if "SEM" in inst else None)
    scope = inst.get(scope_key, {}) if scope_key else {}
    data = np.asarray(signal["data"])
    ax = image_axes(signal)
    if kind(signal) in ("spectrum", "spectrum line"):
        ax = [a for a in signal["axes"] if _is_energy(a)] or signal_axes(signal)
    date = _get(md, "General", "date")
    when = f"{date}T{_get(md, 'General', 'time')}" if date and _get(md, "General", "time") else date
    exposure = (_get(scope, "Detector", "Camera", "exposure") or _get(scope, "Detector", "EDS", "live_time")
                or _get(scope, "dwell_time"))

    def num(v):
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    out = {
        "microscope": {"name": scope.get("microscope"), "voltage": num(scope.get("beam_energy")),
                       "magnification": num(scope.get("magnification")), "camera_length": num(scope.get("camera_length")),
                       "spot_size": None, "type": scope_key},
        "sample": {"name": _get(md, "General", "title") or (path.stem if path else None),
                   "description": _get(md, "Sample", "description"), "preparation": None},
        "acquisition": {"date": when, "operator": _get(md, "General", "authors"), "exposure_time": num(exposure),
                        "tilt_angle": num(_get(scope, "Stage", "tilt_alpha")), "defocus": None},
        "image": {"width": int(ax[-1]["size"]) if ax else None,
                  "height": int(ax[-2]["size"]) if len(ax) >= 2 else None,
                  "bit_depth": int(data.dtype.itemsize * 8), "pixel_size": pixel_size_nm(signal)},
        "processing": {"software": None, "operations": None},
        "signal": {"kind": kind(signal), "type": _get(md, "Signal", "signal_type") or None,
                   "shape": list(data.shape), "dtype": str(data.dtype),
                   "axes": [{k: _plain(a.get(k)) for k in ("name", "size", "scale", "offset", "units", "navigate")}
                            for a in signal["axes"]]},
        "_source": {"file": path.name if path else None, "reader": reader,
                    "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
    return out
