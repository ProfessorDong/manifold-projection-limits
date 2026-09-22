"""Minimal TFRecord + protobuf reader (no TensorFlow).

TFRecord: [uint64 len][uint32 crc][data][uint32 crc]  (CRCs not verified)
Payload : tf.train.Example -> Features -> map<string,Feature>
          DICHASUS stores tensors as serialized TensorProto inside a bytes feature.
"""
import struct, numpy as np

def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7f) << s
        if not (x & 0x80): return r, i
        s += 7

def parse_fields(b, start=0, end=None):
    """Generic protobuf wire parser -> {field_number: [raw values]}."""
    if end is None: end = len(b)
    i = start; out = {}
    while i < end:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if   wt == 0: v, i = _varint(b, i)
        elif wt == 1: v = b[i:i+8]; i += 8
        elif wt == 2:
            ln, i = _varint(b, i); v = b[i:i+ln]; i += ln
        elif wt == 5: v = b[i:i+4]; i += 4
        else: raise ValueError(f"unsupported wire type {wt}")
        out.setdefault(fn, []).append(v)
    return out

_DT = {1: np.float32, 2: np.float64, 3: np.int32, 9: np.int64}

def parse_tensorproto(b):
    f = parse_fields(b)
    dtype = f[1][0] if 1 in f else 1
    shape = []
    if 2 in f:
        for dim in parse_fields(f[2][0]).get(2, []):
            shape.append(parse_fields(dim).get(1, [0])[0])
    if 4 in f:                                    # tensor_content: raw LE bytes
        arr = np.frombuffer(f[4][0], dtype=_DT.get(dtype, np.float32))
    elif 5 in f:                                  # float_val (packed)
        arr = np.frombuffer(f[5][0], dtype=np.float32)
    else:
        raise ValueError("no tensor payload")
    return arr.reshape(shape) if shape else arr

def parse_example(b):
    """-> dict name -> raw bytes / float array / int list"""
    ex = parse_fields(b)
    feats = parse_fields(ex[1][0])                 # Features
    out = {}
    for entry in feats.get(1, []):                 # repeated map entries
        e = parse_fields(entry)
        name = e[1][0].decode()
        feat = parse_fields(e[2][0])
        if   1 in feat: out[name] = parse_fields(feat[1][0]).get(1, [b''])[0]     # bytes_list
        elif 2 in feat: out[name] = np.frombuffer(parse_fields(feat[2][0])[1][0], np.float32)
        elif 3 in feat:
            vals = parse_fields(feat[3][0]).get(1, [])
            out[name] = np.array(vals, dtype=np.int64)
    return out

def iter_records(path, max_records=None, max_bytes=None):
    """Yield parsed examples from a (possibly truncated) tfrecords file."""
    n = 0
    with open(path, 'rb') as fh:
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8: return
            ln = struct.unpack('<Q', hdr)[0]
            if fh.read(4) is None: return
            data = fh.read(ln)
            if len(data) < ln: return              # truncated tail -> stop cleanly
            fh.read(4)
            try:
                yield parse_example(data)
            except Exception:
                return
            n += 1
            if max_records and n >= max_records: return
            if max_bytes and fh.tell() >= max_bytes: return
