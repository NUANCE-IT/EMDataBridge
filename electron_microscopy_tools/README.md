# EMDataBridge

A bridge between electron microscopy data formats, and the researcher identity shared by the NUANCE tools.

Created by **Roberto dos Reis ([@rmsreis](https://github.com/rmsreis))** · NUANCE Center, Northwestern University

1. **Identity**: ORCID-validated researcher identity with an Ed25519 signing key. NUScope, NUScope-TOMO and
   NUANCE-MCP use it to sign data, so a file can prove who acquired, processed or approved it
2. **Format Standardizer**: Read DM, EMD, SER, MRC, TIFF, Bruker, JEOL and other files and map their metadata to one schema
3. **Format Translator**: Convert between formats without touching the pixels or the calibration
4. **Metadata Discovery**: Automatically discover and extract self-describing data organization from electron microscopy files

## Installation

```bash
pip install EMDataBridge                 # identity layer (light: numpy, pyyaml, cryptography)
pip install "EMDataBridge[tools,io]"     # + standardize / translate / discover
pip install "EMDataBridge[all]"          # + the optional language-model step (transformers)
```

## Identity: ORCID and signing key

```bash
emdb identity create "Your Name" --orcid 0000-0002-1825-0097   # once; the ORCID check digit is validated
emdb identity show
emdb identity export me.json          # public part only: share it with collaborators or the facility
emdb identity trust colleague.json    # trust someone else's key
emdb identity list
emdb identity check-orcid 0000-0002-1825-0097
```

```python
from EMDataBridge import identity

me = identity.me()                                    # name, ORCID, key fingerprint
entry = identity.sign({"seal": "<sha256>"}, role="author")
identity.verify({"seal": "<sha256>"}, entry)          # Check(valid=True, trusted=True, signer={...})
```

- The private key is stored in `~/.emdatabridge/identity/` (or `$EMDATABRIDGE_HOME`), readable only by you. Set
  `EMDATABRIDGE_KEY_PASSPHRASE` to keep it encrypted. Back it up: a lost key cannot sign again, but everything it
  signed stays verifiable, because each signature carries the public key.
- A signature covers the signed claims and the signer record (name, ORCID, key fingerprint, role, time), so
  none of them can be changed afterwards.
- Keys created by NUScope 0.3 (`~/.nuscope/keys`) are imported automatically on first use. The originals are left in place.

## Reading, standardizing and translating files

```bash
emdb formats                                    # what is supported: read, write or planned
emdb standardize scan.dm4                       # metadata in the standard schema (JSON; -f yaml)
emdb standardize scan.dm4 -o scan.json
emdb translate scan.dm4 scan.tif                # pixels and calibration written as they are
emdb translate raw/ out/ -b -f hspy             # a whole folder
```

```python
from EMDataBridge import EMDataStandardizer, EMFormatTranslator

out = EMDataStandardizer().standardize("scan.dm4")
out["data"]                     # the array, exactly as stored
out["metadata"]["microscope"]   # {'name': 'JEOL COM', 'voltage': 300.0, 'magnification': 500000.0, ...}
out["metadata"]["image"]        # {'width': 2048, 'height': 2048, 'bit_depth': 16, 'pixel_size': 0.288}  (nm)
out["metadata"]["signal"]       # kind (image, spectrum, spectrum image, image stack, 4D dataset), shape, axes

EMFormatTranslator().translate("scan.dm4", "scan.tif")   # also writes scan.tif.emdb.json
```

- **Reading** uses [RosettaSciIO](https://hyperspy.org/rosettasciio/), the HyperSpy file readers, without
  needing HyperSpy. Instrument values are copied out of the vendor tags the same way HyperSpy does.
- **The schema** maps each field from a named place in the file. A value the file doesn't record is `null`:
  an unknown voltage is not 0 kV. Units: kV, mm, s, degrees, nm. `exposure_time` is the camera exposure, or
  the dwell time for scanned images.
- **Translating never changes the data.** No rescaling and no change of number type: if the target can't hold
  the data as it is (MRC can't store float64; an MSA file holds one spectrum), you get an error, not a silent
  conversion. Formats that can't carry all the metadata get a `<output>.emdb.json` beside them with the full
  standardized metadata. HyperSpy and HDF5 keep everything.
- **Language model (optional, off by default):** `--llm mock|llama` lets a model propose values for vendor
  fields the explicit mapping misses. It only fills gaps, and what it adds is recorded under `_llm_provenance`.

## Supported formats

| Format | Extensions | Read | Write |
|---|---|---|---|
| Gatan Digital Micrograph | .dm3 .dm4 | ✓ | (no open-source writer exists) |
| Thermo Fisher Velox / Berkeley EMD | .emd | ✓ | ✓ (Berkeley EMD) |
| Thermo Fisher TIA | .ser .emi | ✓ | |
| EMSA/MSA spectra | .msa .emsa .ems .mas | ✓ | ✓ (single spectra) |
| MRC, including tomography stacks | .mrc .mrcs .ali .rec .st | ✓ | ✓ |
| TIFF (Thermo Fisher/FEI, Zeiss, TVIPS, Olympus, JEOL SightX, Hamamatsu metadata) | .tif .tiff | ✓ | ✓ |
| HyperSpy | .hspy | ✓ | ✓ |
| HDF5 | .h5 .hdf5 | ✓ | ✓ |
| Bruker | .bcf .spx | ✓ | |
| JEOL | .asw .pts .map .eds | ✓ | |
| EDAX | .spd .spc | ✓ | |
| NanoMegas blockfile, TVIPS, Quantum Detectors MIB | .blo .tvips .mib | ✓ | |
| EM (TOM/EMAN), Hitachi, Oxford Instruments AZtec | .em .hds .h5oina .oip | planned | planned |

Hitachi and Oxford need sample files to build and test against: if you have some, open an issue.

## Requirements

Python 3.10 or newer. Dependencies are in `pyproject.toml`: the identity layer needs only numpy, pyyaml and
cryptography; `[io]` adds RosettaSciIO, mrcfile and h5py; `[tools]` adds what discovery and benchmarks use.

## Benchmarking

The package includes benchmarking tools to measure performance of format conversions and metadata extraction.

### Running Benchmarks

```bash
# Run the sample benchmark suite
python run_benchmarks.py

# Run specific benchmarks using the benchmark module
python -m EMDataBridge.benchmark standardize --files sample_data/*.tif --output-dir benchmark_results
python -m EMDataBridge.benchmark translate --files sample_data/*.dm4 --formats tiff hdf5 mrc --output-dir benchmark_results
python -m EMDataBridge.benchmark batch --input-dir sample_data --formats tiff emd hspy --output-dir benchmark_results

# Run a full benchmark suite
python -m EMDataBridge.benchmark full --files sample_data/*.* --formats tiff hdf5 emd hspy --input-dir sample_data --output-dir benchmark_results
```

### Benchmark Reports

Benchmark results are saved as JSON files in the output directory, along with visualizations showing performance metrics for different formats and operations.
