# Measured channel data

**Nothing from the DICHASUS datasets is redistributed in this repository** — neither
the channel recordings nor the small metadata files. We do not hold a copy of their
licence terms, so we do not republish any of their content. Obtain them yourself
from the sources below; `fetch_data.py --verify` will confirm you have the same
files we used, by checksum.

| Used as | Collection | DOI |
|---|---|---|
| industrial hall, distributed arrays | `dichasus-cf0x` (record `cf06`) | `10.18419/darus-2854` |
| lab room, co-located array | `dichasus-015x` (record `0152`) | `10.18419/darus-2202` |

Project site: <https://dichasus.inue.uni-stuttgart.de/>. Please cite the dataset
records and the sounder paper:

> F. Euchner, M. Gauger, S. Dörner and S. ten Brink, "A distributed massive MIMO
> channel sounder for 'Big CSI Data'-driven machine learning," in *Proc. Int. ITG
> Workshop Smart Antennas (WSA)*, 2021, pp. 289–294.

Check the licence on each DaRUS landing page before redistributing anything derived
from the recordings.

## What you need to put in this directory

**Channel recordings** (downloaded by `fetch_data.py`, given the TFRecord URL from
the DaRUS landing page):

| File | Content | SHA-256 |
|---|---|---|
| `cf06_head.bin` | first 67 108 864 bytes of the `cf06` TFRecord, 255 records | `02ed3d25…bbd7d` |
| `d0152_head.bin` | first 67 108 864 bytes of the `0152` TFRecord, 255 records | `9ca7cd68…5d0b3` |

**Metadata** published alongside the datasets, copied in by hand:

| File | Content | SHA-256 |
|---|---|---|
| `spec.json` | `cf06` array geometry: element assignments, spacings, positions, orientations, bandwidth | `9a1d3c1c…50b5d` |
| `spec015.json` | `0152` array geometry, same fields | `98f3573c…e0a46` |
| `offsets_cf06.json` | published per-antenna carrier-phase (`cpo`) and sampling-time (`sto`) estimates | `3ed57c2c…6db82` |
| `offsets_015x.json` | the same for `0152` | `2544dcd5…e930bf` |

Full hashes are in `fetch_data.py`. Run `python data/fetch_data.py --verify` from the
repository root; the scripts in `sim/` that need these files name them directly if
they are absent.

Written by the scripts and not versioned: `*.npy` caches.

## Exactly what the paper uses

- **Byte range.** The first 67 108 864 bytes of each TFRecord, obtained with an HTTP
  range request. Any longer prefix also works; the readers stop at the last complete
  record.
- **Records.** Every eighth record of the first 256, that is indices 0, 8, 16, …, 248,
  giving the 32 snapshots of Table III. The hall set traverses 4.7 m over 13.2 s; the
  lab set spans 0.16 m and is not in chronological file order, which the paper states.
- **Sub-array.** One 2×4 uniform rectangular sub-array of 0.118 m spacing
  (0.501 λ at 1.272 GHz), so that `N_r = 8` and `N = 8192` for both datasets. The hall
  uses the assignment in `spec.json`; the lab uses the top-left 2×4 block of the 4×8
  array in `spec015.json`.
- **Offset compensation.** The hall applies both the carrier-phase and the
  sampling-time offset; the lab applies the sampling-time offset alone. That is the
  combination minimising the order-4 specular residual for each dataset — 0.420 and
  0.419 respectively. `sim/verify_offset_convention.py` prints every combination so
  the choice is auditable.
