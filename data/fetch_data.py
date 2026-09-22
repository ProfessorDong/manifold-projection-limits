"""Fetch and verify the measured channel prefixes the paper uses.

The DICHASUS recordings are third-party and are not redistributed with this code.
This script pulls exactly the byte range the paper used and checks it against the
recorded SHA-256, so a reader can confirm they are working from the same bytes.

The per-file download URLs on DaRUS are not hard-coded here on purpose: they are
tied to the repository's internal file identifiers and change independently of the
dataset DOI. Take the URL of the TFRecord you want from the DaRUS landing page for
the DOI listed in README.md and pass it with --url.

    python data/fetch_data.py --dataset cf06 --url <TFRecord URL>
    python data/fetch_data.py --verify

Any prefix at least as long as the recorded size works; the readers in sim/ stop
at the last complete record.
"""
import argparse, hashlib, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBYTES = 67_108_864                      # 64 MiB, = 255 complete records
EXPECT = {
    'cf06': ('cf06_head.bin',
             '02ed3d258719bbc46325508be3c38fb62563af6eb52cc51b9e4c3458d2cbbd7d'),
    '0152': ('d0152_head.bin',
             '9ca7cd6851f1af1f6dc21196e2cf2c25445c4ce24dcf32e0d58eb3ce6455d0b3'),
}

# Small metadata files published with the datasets: array geometry and the
# per-antenna reference-transmitter offset estimates.  Not redistributed here
# either; obtain them from the same source and drop them in this directory.
# Hashes are given so you can confirm you have the same files we used.
METADATA = {
    'spec.json':         '9a1d3c1c8edb39691c2c28f1dba61c85d70fce95aa38bcfb2669c822cb450b5d',
    'spec015.json':      '98f3573cff1050b36e3b7c5fa0b2878501e2bc4840df60d05787a95986ae0a46',
    'offsets_cf06.json': '3ed57c2c8a7603cdcb8f6a144e013722508dc5256130e0f23053105a33a6db82',
    'offsets_015x.json': '2544dcd5a3a04b4e492e57e93c808e01bb71dc00374415ca7fb0b6978be930bf',
}

def sha256(path, nbytes=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        data = f.read() if nbytes is None else f.read(nbytes)
    h.update(data)
    return h.hexdigest(), len(data)

def records(path):
    sys.path.insert(0, str(HERE.parent / 'sim'))
    from tfrecord_reader import iter_records
    return sum(1 for _ in iter_records(str(path), max_records=10_000_000))

def verify(name):
    fn, want = EXPECT[name]
    p = HERE / fn
    if not p.exists():
        print(f"  {fn:16s} MISSING  (run with --dataset {name} --url ...)")
        return False
    got, n = sha256(p, NBYTES)
    ok = (n == NBYTES and got == want)
    print(f"  {fn:16s} {n} bytes  sha256 {'OK' if got == want else 'MISMATCH'}"
          f"  records {records(p)}")
    if not ok and n != NBYTES:
        print(f"    (expected at least {NBYTES} bytes; a longer prefix is fine, "
              f"but the checksum is over the first {NBYTES})")
    return ok

def verify_metadata(fn):
    p = HERE / fn
    if not p.exists():
        print(f"  {fn:20s} MISSING")
        return False
    got, _ = sha256(p)
    print(f"  {fn:20s} sha256 {'OK' if got == METADATA[fn] else 'MISMATCH'}")
    return got == METADATA[fn]

def download(name, url):
    fn, _ = EXPECT[name]
    out = HERE / fn
    req = urllib.request.Request(url, headers={'Range': f'bytes=0-{NBYTES-1}'})
    with urllib.request.urlopen(req) as r:
        if r.status != 206:
            print(f"warning: server returned {r.status}, not 206 Partial Content; "
                  f"the whole file may be downloading", file=sys.stderr)
        data = r.read(NBYTES)
    out.write_bytes(data)
    print(f"wrote {out} ({len(data)} bytes)")
    verify(name)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=sorted(EXPECT))
    ap.add_argument('--url', help='TFRecord URL from the DaRUS landing page')
    ap.add_argument('--verify', action='store_true', help='check what is already here')
    a = ap.parse_args()
    if a.verify or not a.dataset:
        print("recordings:")
        ok = all([verify(k) for k in sorted(EXPECT)])
        print("metadata (array geometry and reference-transmitter offsets):")
        ok = all([verify_metadata(k) for k in sorted(METADATA)]) and ok
        if not ok:
            print("\nMissing files are third-party and are not redistributed with this")
            print("code. See README.md in this directory for the datasets and their DOIs.")
        sys.exit(0 if ok else 1)
    if not a.url:
        ap.error('--url is required with --dataset; see README.md')
    download(a.dataset, a.url)
