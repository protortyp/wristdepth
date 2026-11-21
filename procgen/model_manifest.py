import blenderproc as bproc  # BlenderProc requires this to be the first line
# isort: skip_file

# List Poly Haven models with their real-world size, for picking tabletop objects.
#
#     blenderproc run model_manifest.py HAVEN_MODELS_DIR OUT.json
#
import json
import sys
from pathlib import Path

import bpy
import numpy as np

bproc.init()
root, out = Path(sys.argv[1]), Path(sys.argv[2])
manifest = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    blends = sorted(d.glob("*.blend"))
    if not blends:
        continue
    try:
        objs = [
            o
            for o in bproc.loader.load_blend(str(blends[0]))
            if isinstance(o, bproc.types.MeshObject)
        ]
    except Exception as e:  # noqa: BLE001  (a broken asset must not stop the scan)
        print("skip", d.name, e)
        continue
    if objs:
        pts = np.concatenate([o.get_bound_box() for o in objs])
        size = (pts.max(0) - pts.min(0)).tolist()
        manifest.append(
            {"name": d.name, "blend": str(blends[0]), "size_m": size, "parts": len(objs)}
        )
        print(d.name, np.round(size, 3))
    for o in bpy.data.objects:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_recursive=True)
out.write_text(json.dumps(manifest, indent=1))
print("models", len(manifest))
