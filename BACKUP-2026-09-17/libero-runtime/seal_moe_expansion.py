"""Record source identities and final artifact hashes for the completed analysis."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path('/data/libero-runtime/samples/moe-joint-expanded-20260915')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    pixel_checks = []
    for path in sorted(ROOT.glob('*.png')):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
        fraction = float(np.any(pixels < 245, axis=-1).mean())
        if fraction < .01 or pixels.std() < 2:
            raise ValueError(f'blank figure: {path}')
        pixel_checks.append(dict(file=path.name, width=pixels.shape[1], height=pixels.shape[0],
                                 nonwhite_fraction=fraction, standard_deviation=float(pixels.std())))
    (ROOT / 'figure-verification.json').write_text(json.dumps(
        dict(pixel_checks=pixel_checks, visually_inspected=True), indent=2) + '\n')
    sources = [Path('/data/libero-runtime') / name for name in (
        'restore_moe_expansion_inputs.py', 'analyze_moe_expansion.py', 'plot_moe_expansion.py',
        'test_moe_expansion.py', 'verify_moe_expansion.py', 'seal_moe_expansion.py')]
    sources.extend([
        Path('/data/coding/v8-methods/moe_joint_patterns.py'),
        Path('/data/coding/robot-whisper-0909/safe&vlaconf/moe_trainfree/routing_dynamics/encoder.py'),
        Path('/data/libero-runtime/samples/moe-joint-patterns-20260915/artifact-hashes.json'),
        Path('/data/libero-runtime/samples/moe-joint-patterns-20260915/source-hashes.json'),
        Path('/data/coding/v8-signal-test-CTUaAJ/summary.json'),
    ])
    (ROOT / 'implementation-hashes.json').write_text(json.dumps(
        {str(path): sha256(path) for path in sources}, indent=2) + '\n')
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name != 'artifact-hashes.json')
    manifest = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    (ROOT / 'artifact-hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(files=len(paths), bytes=sum(p.stat().st_size for p in paths),
                          root=str(ROOT), figures=pixel_checks), indent=2))


if __name__ == '__main__':
    main()
