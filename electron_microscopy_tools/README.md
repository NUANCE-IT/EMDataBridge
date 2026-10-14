# EMDataBridge

A bridge between electron microscopy data formats, and the researcher identity shared by the NUANCE tools.

Created by **Roberto dos Reis ([@rmsreis](https://github.com/rmsreis))** · NUANCE Center, Northwestern University

1. **Identity**: ORCID-validated researcher identity with an Ed25519 signing key. NUScope, NUScope-TOMO and
   NUANCE-MCP use it to sign data, so a file can prove who acquired, processed or approved it
2. **Format Standardizer**: Standardize various electron microscopy data formats to a common schema
3. **Format Translator**: Convert between different electron microscopy data formats
4. **Metadata Discovery**: Automatically discover and extract self-describing data organization from electron microscopy files

## Installation

```bash
pip install EMDataBridge                 # identity layer (light: numpy, pyyaml, cryptography)
pip install "EMDataBridge[tools,io]"     # + standardize / translate / discover
pip install "EMDataBridge[all]"          # + transformers and HyperSpy
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

## Usage

See the documentation for each tool in their respective modules:

- `standardizer.py` - Standardize data formats
- `translator.py` - Translate between formats
- `discovery.py` - Discover self-describing data organization

## Supported Formats

- MRC (Medical Research Council)
- EM (Electron Microscopy)
- DM3/DM4 (Digital Micrograph)
- TIF/TIFF with metadata
- HDF5/Zarr-based formats
- MSA (EMSA format)
- EMD (Thermo Fisher Scientific's electron microscopy data format, formerly FEI)
- SER (Thermo Fisher Scientific's TIA series data format, formerly FEI)
- SerialEM formats
- Oxford Instruments formats (AZtec, INCA)
- Hitachi formats (.hds, .tif with metadata)
- Thermo Fisher Scientific formats (formerly FEI)

## Requirements

See `requirements.txt` for dependencies.

## Benchmarking

The package includes benchmarking tools to measure performance of format conversions and metadata extraction.

### Running Benchmarks

```bash
# Run the sample benchmark suite
python run_benchmarks.py

# Run specific benchmarks using the benchmark module
python benchmark.py standardize --files sample_data/*.tif --output-dir benchmark_results
python benchmark.py translate --files sample_data/*.dm4 --formats tiff hdf5 mrc --output-dir benchmark_results
python benchmark.py batch --input-dir sample_data --formats tiff emd ser --output-dir benchmark_results

# Run a full benchmark suite
python benchmark.py full --files sample_data/*.* --formats tiff hdf5 emd ser --input-dir sample_data --output-dir benchmark_results
```

### Benchmark Reports

Benchmark results are saved as JSON files in the output directory, along with visualizations showing performance metrics for different formats and operations.
