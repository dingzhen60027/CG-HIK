"""Read-only DROID data helpers. Never import/instantiate its robot environment.

TFRecord/protobuf parsing only extracts episode identity and low-dimensional
features. Encoded images are skipped, never decoded. Raw HDF5 is authoritative
for action-time state and timestamps when available.
"""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np


def varint(data, pos):
    value = shift = 0
    while True:
        b = data[pos]; pos += 1
        value |= (b & 127) << shift
        if b < 128:
            return value, pos
        shift += 7
        if shift > 70:
            raise ValueError('bad protobuf varint')


def fields(data):
    """Standard protobuf wire fields, with length-delimited zero-copy views."""
    data = memoryview(data); pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos); wire = tag & 7
        if wire == 0:
            value, pos = varint(data, pos)
        elif wire in (1, 5):
            n = 8 if wire == 1 else 4
            value = data[pos:pos+n]; pos += n
        elif wire == 2:
            n, pos = varint(data, pos)
            value = data[pos:pos+n]; pos += n
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')
        if pos > len(data):
            raise ValueError('truncated protobuf')
        yield tag >> 3, wire, value


def example_features(data):
    features = next(v for k, _, v in fields(data) if k == 1)
    result = {}
    for _, _, entry in fields(features):
        pair = {k:v for k, _, v in fields(entry)}
        key = bytes(pair[1]).decode()
        if 'image' in key:
            continue
        kind, _, payload = next(fields(pair[2]))
        if kind == 1:
            value = [bytes(v) for _, _, v in fields(payload)]
        elif kind == 2:
            value = np.frombuffer(next(fields(payload))[2], '<f4').tolist()
        elif kind == 3:
            packed = next(fields(payload))[2]; p = 0; value = []
            while p < len(packed):
                v, p = varint(packed, p); value.append(v)
        else:
            raise ValueError(kind)
        result[key] = value
    return result


def read_tfrecord(path):
    # Source object MD5/SHA256 is checked by downloader; preserve TFRecord CRC
    # bytes in the original local shard (not claiming a separate CRC32C check).
    with Path(path).open('rb') as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) != 12:
                raise ValueError('truncated TFRecord header')
            n = struct.unpack('<Q', head[:8])[0]
            data = f.read(n); crc = f.read(4)
            if len(data) != n or len(crc) != 4:
                raise ValueError('truncated TFRecord payload')
            yield example_features(data)


def fetch_object(obj, directory):
    """Only the explicitly supplied public object, with checksum validation."""
    import base64
    import requests
    from urllib.parse import quote
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    name = obj['name']; path = directory / name.rsplit('/', 1)[-1]
    if not path.exists():
        url = 'https://storage.googleapis.com/gresearch/' + quote(name, safe='/')
        response = requests.get(url, stream=True, timeout=(20, 90))
        response.raise_for_status()
        with path.open('xb') as f:
            for block in response.iter_content(1024*1024):
                f.write(block)
    digest = hashlib.md5(); sha = hashlib.sha256(); size = 0
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            digest.update(block); sha.update(block); size += len(block)
    assert size == int(obj['size']), (name, size, obj['size'])
    assert base64.b64encode(digest.digest()).decode() == obj['md5Hash'], name
    return dict(object=name, bytes=size, sha256=sha.hexdigest(),
                md5_verified=True, path=str(path))


def identity_index(shard):
    records = []
    for index, data in enumerate(read_tfrecord(shard['path'])):
        identity = {k:[v.decode() for v in vs] for k, vs in data.items()
                    if k.startswith('episode_metadata/')}
        records.append(dict(shard=shard['object'], record=index, metadata=identity,
                            fields={k:len(v) for k,v in data.items()}))
    return records


RAW_FIELDS = ['action/cartesian_position', 'action/joint_position',
    'action/robot_state/joint_positions', 'action/robot_state/cartesian_position',
    'observation/robot_state/joint_positions', 'observation/robot_state/cartesian_position',
    'observation/timestamp/control/step_start', 'observation/timestamp/control/control_start',
    'observation/timestamp/control/step_end']


def raw_schema(path):
    import h5py
    with h5py.File(path, 'r') as f:
        missing = [k for k in RAW_FIELDS if k not in f]
        if missing:
            return dict(eligible=False, reason='missing_fields', missing=missing)
        data = {k:f[k][()] for k in RAW_FIELDS}
        n=len(data[RAW_FIELDS[0]])
        shapes=[(n,6),(n,7),(n,7),(n,6),(n,7),(n,6),(n,),(n,),(n,)]
        good=all(x.shape==shape and np.isfinite(x).all() for x,shape in zip(data.values(),shapes))
        times=data['observation/timestamp/control/step_start']
        ordered=bool(np.all(np.diff(times)>0))
        return dict(eligible=bool(good and n>1 and ordered),frames=n,
            reason='usable' if good and n>1 and ordered else 'shape_finiteness_or_nonmonotonic_time',
            source_version=str(f.attrs.get('version_number','unknown')),
            fields={k:list(v.shape) for k,v in data.items()},
            robot_serial=str(f.attrs.get('robot_serial_number','unknown')),
            step_interval_ms_quantiles=np.percentile(np.diff(times),[0,50,95,100]).tolist(),
            nonmonotonic_step_count=int(np.sum(np.diff(times)<=0)))
