"""Reading, mapping and writing real files: pixels must come back identical, and unknown is None."""
import json

import numpy as np
import pytest

pytest.importorskip("rsciio")

from EMDataBridge import formats
from EMDataBridge.standardizer import EMDataStandardizer
from EMDataBridge.translator import EMFormatTranslator

TEM = {"General": {"title": "grain boundary", "date": "2026-11-04", "time": "10:15:00", "authors": "R. dos Reis"},
       "Signal": {"signal_type": "", "quantity": "Intensity"},
       "Acquisition_instrument": {"TEM": {"beam_energy": 200.0, "magnification": 50000.0, "microscope": "JEOL ARM200F",
                                          "camera_length": 80.0, "Stage": {"tilt_alpha": 12.5},
                                          "Detector": {"Camera": {"exposure": 0.5}}}}}


def image(dtype=np.float32):
    data = (np.arange(12 * 16).reshape(12, 16) * 1.37).astype(dtype)
    axes = [{"name": "y", "size": 12, "scale": 0.5, "offset": 0.0, "units": "nm", "navigate": False},
            {"name": "x", "size": 16, "scale": 0.5, "offset": 0.0, "units": "nm", "navigate": False}]
    return {"data": data, "axes": axes, "metadata": json.loads(json.dumps(TEM)), "original_metadata": {"vendor": {"k": 1}}}


@pytest.fixture
def hspy(tmp_path):
    p = tmp_path / "scan.hspy"
    formats.write(image(), p)
    return p


def test_hyperspy_file_maps_every_field(hspy):
    out = EMDataStandardizer().standardize(hspy)
    md = out["metadata"]
    assert md["microscope"] == {"name": "JEOL ARM200F", "voltage": 200.0, "magnification": 50000.0,
                                "camera_length": 80.0, "spot_size": None, "type": "TEM"}
    assert md["acquisition"]["date"] == "2026-11-04T10:15:00"
    assert md["acquisition"]["operator"] == "R. dos Reis"
    assert md["acquisition"]["exposure_time"] == 0.5 and md["acquisition"]["tilt_angle"] == 12.5
    assert md["image"] == {"width": 16, "height": 12, "bit_depth": 32, "pixel_size": 0.5}
    assert md["signal"]["kind"] == "image" and md["signal"]["dtype"] == "float32"
    assert md["_source"]["reader"] == "rsciio.hspy"
    assert np.array_equal(out["data"], image()["data"])


def test_unknown_is_none_not_zero(tmp_path):
    p = tmp_path / "bare.tif"
    formats.write({"data": np.ones((8, 8), np.uint16)}, p)
    md = EMDataStandardizer().standardize(p)["metadata"]
    assert md["microscope"]["voltage"] is None and md["microscope"]["magnification"] is None
    assert md["microscope"]["name"] is None and md["acquisition"]["date"] is None


def test_translate_to_tiff_keeps_pixels_and_writes_sidecar(hspy, tmp_path):
    out = tmp_path / "scan.tif"
    r = EMFormatTranslator().translate(hspy, out)
    back = formats.read(out)[0]
    assert back["data"].dtype == np.float32                       # no rescaling to 8-bit
    assert np.array_equal(back["data"], image()["data"])
    assert formats.pixel_size_nm(back) == pytest.approx(0.5)
    side = json.loads((tmp_path / "scan.tif.emdb.json").read_text())
    assert side["microscope"]["voltage"] == 200.0 and side["_translation"]["from"] == "scan.hspy"
    assert r["sidecar"].endswith(".emdb.json")


def test_translate_to_hyperspy_needs_no_sidecar(hspy, tmp_path):
    r = EMFormatTranslator().translate(hspy, tmp_path / "copy.hspy")
    assert "sidecar" not in r
    assert formats.read(tmp_path / "copy.hspy")[0]["original_metadata"]["vendor"]["k"] == 1


def test_mrc_round_trip_keeps_voxel_size(hspy, tmp_path):
    EMFormatTranslator().translate(hspy, tmp_path / "scan.mrc")
    out = EMDataStandardizer().standardize(tmp_path / "scan.mrc")
    assert np.array_equal(out["data"], image()["data"])
    assert out["metadata"]["image"]["pixel_size"] == pytest.approx(0.5)


def test_mrc_refuses_float64_instead_of_converting(tmp_path):
    with pytest.raises(formats.FormatError, match="float64"):
        formats.write(image(np.float64), tmp_path / "x.mrc")


def test_emd_and_hdf5_round_trips(hspy, tmp_path):
    for name in ("scan.emd", "scan.h5"):
        EMFormatTranslator().translate(hspy, tmp_path / name)
        out = EMDataStandardizer().standardize(tmp_path / name)
        assert np.array_equal(out["data"], image()["data"]), name
        assert out["metadata"]["image"]["pixel_size"] == pytest.approx(0.5), name
    assert EMDataStandardizer().standardize(tmp_path / "scan.h5")["metadata"]["microscope"]["voltage"] == 200.0


def test_generic_hdf5_uses_the_largest_dataset(tmp_path):
    import h5py
    p = tmp_path / "other.h5"
    with h5py.File(p, "w") as f:
        f["small"] = np.zeros(3)
        f.create_group("entry")["frame"] = np.arange(64, dtype=np.int16).reshape(8, 8)
    out = EMDataStandardizer().standardize(p)
    assert out["data"].shape == (8, 8) and out["metadata"]["sample"]["name"] == "entry/frame"


def test_msa_spectrum(tmp_path):
    spec = {"data": np.linspace(0, 1, 100),
            "axes": [{"name": "Energy", "size": 100, "scale": 0.01, "offset": 0.0, "units": "keV", "navigate": False}],
            "metadata": {**TEM, "Signal": {"signal_type": "EDS_TEM", "quantity": "X-rays (Counts)"}}}
    formats.write(spec, tmp_path / "eds.msa")
    md = EMDataStandardizer().standardize(tmp_path / "eds.msa")["metadata"]
    assert md["signal"]["kind"] == "spectrum" and md["signal"]["type"] == "EDS_TEM"
    assert md["image"]["width"] == 100 and md["image"]["height"] is None
    assert md["image"]["pixel_size"] is None                       # keV is not a length
    assert md["microscope"]["voltage"] == 200.0
    with pytest.raises(formats.FormatError, match="single spectrum"):
        formats.write(image(), tmp_path / "img.msa")


def test_honest_about_what_is_not_supported(tmp_path):
    with pytest.raises(formats.FormatError, match="no open-source library writes DM"):
        EMFormatTranslator().translate(tmp_path / "a.hspy", tmp_path / "a.dm4")
    (tmp_path / "x.hds").write_bytes(b"\0")
    with pytest.raises(NotImplementedError, match="Hitachi"):
        EMDataStandardizer().standardize(tmp_path / "x.hds")
    with pytest.raises(ValueError, match="Unsupported"):
        EMDataStandardizer().standardize(tmp_path / "x.xyz")
    status = {f["name"]: f["status"] for f in EMFormatTranslator.list_supported_formats()}
    assert status["Gatan Digital Micrograph"] == "read"
    assert status["TIFF"] == "read, write"
    assert status["Hitachi"] == "planned" and status["Oxford Instruments AZtec"] == "planned"


def test_stack_kind(tmp_path):
    data = np.zeros((3, 4, 5), np.float32)
    axes = [{"name": "z", "size": 3, "scale": 1.0, "offset": 0.0, "units": "", "navigate": True},
            {"name": "y", "size": 4, "scale": 2.0, "offset": 0.0, "units": "µm", "navigate": False},
            {"name": "x", "size": 5, "scale": 2.0, "offset": 0.0, "units": "µm", "navigate": False}]
    formats.write({"data": data, "axes": axes}, tmp_path / "s.hspy")
    md = EMDataStandardizer().standardize(tmp_path / "s.hspy")["metadata"]
    assert md["signal"]["kind"] == "image stack"
    assert md["image"]["pixel_size"] == pytest.approx(2000.0)     # µm -> nm


def test_vendor_mapping_is_applied_without_hyperspy():
    sig = {"data": np.zeros((4, 4)), "axes": [], "metadata": {"Acquisition_instrument": {"TEM": {"microscope": None}}},
           "original_metadata": {"ImageList": {"TagGroup0": {"ImageTags": {
               "Microscope Info": {"Voltage": 300000.0, "Name": "JEOL COM", "Bad": "x"},
               "Session Info": {"Operator": "alp"}}}}},
           "mapping": {"ImageList.TagGroup0.ImageTags.Microscope_Info.Voltage": ("Acquisition_instrument.TEM.beam_energy", lambda v: v / 1e3),
                       "ImageList.TagGroup0.ImageTags.Session Info.Operator": ("General.authors", None),
                       "ImageList.TagGroup0.ImageTags.Microscope Info.Bad": ("General.x", lambda v: 1 / 0),
                       "ImageList.TagGroup0.ImageTags.Missing": ("General.y", None)}}
    md = formats.apply_mapping(sig)["metadata"]
    assert md["Acquisition_instrument"]["TEM"]["beam_energy"] == 300.0            # 'Microscope_Info' == 'Microscope Info'
    assert md["Acquisition_instrument"]["TEM"]["microscope"] == "JEOL COM"       # an empty value gets filled
    assert md["General"] == {"authors": "alp"}                                   # a failing converter is skipped
    assert "mapping" not in sig


def test_energy_first_spectrum_image_is_a_spectrum_image():
    sig = {"data": np.zeros((50, 6, 4), np.float32),
           "axes": [{"name": "Energy loss", "size": 50, "scale": 0.03, "offset": 0, "units": "eV", "navigate": True},
                    {"name": "y", "size": 6, "scale": 0.0167, "offset": 0, "units": "µm", "navigate": False},
                    {"name": "x", "size": 4, "scale": 0.0167, "offset": 0, "units": "µm", "navigate": False}]}
    md = formats.to_schema(formats.complete(sig))
    assert md["signal"]["kind"] == "spectrum image"
    assert (md["image"]["width"], md["image"]["height"]) == (4, 6) and md["image"]["pixel_size"] == pytest.approx(16.7)
