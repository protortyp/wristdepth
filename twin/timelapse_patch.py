"""Patch a gsplat example trainer to render fixed cameras every N steps (training timelapse).

    python timelapse_patch.py simple_trainer.py           # patches in place
    TIMELAPSE_EVERY=100 TIMELAPSE_CAMS=cams.json python simple_trainer.py ...

cams.json: [{"c2w": 4x4 OpenCV camera-to-world in the training frame, "K": 3x3, "w": W, "h": H}].
Frames go to <result_dir>/timelapse/<step>.jpg, cameras side by side at 480 px height.
"""

import sys
from pathlib import Path

METHOD = """
    @torch.no_grad()
    def _timelapse(self, step: int) -> None:
        import json as _json
        cams = _json.loads(open(os.environ["TIMELAPSE_CAMS"]).read())
        tiles = []
        for cam in cams:
            c2w = torch.tensor(cam["c2w"], dtype=torch.float32, device=self.device)[None]
            K = torch.tensor(cam["K"], dtype=torch.float32, device=self.device)[None]
            out = self.rasterize_splats(
                camtoworlds=c2w, Ks=K, width=cam["w"], height=cam["h"],
                sh_degree=self.cfg.sh_degree, near_plane=0.005, far_plane=self.cfg.far_plane,
            )
            rgb = torch.clamp(out[0][0, ..., :3], 0.0, 1.0)
            scale = 480 / cam["h"]
            rgb = torch.nn.functional.interpolate(
                rgb.permute(2, 0, 1)[None], scale_factor=scale, mode="area"
            )[0].permute(1, 2, 0)
            tiles.append((rgb.cpu().numpy() * 255).astype("uint8"))
        os.makedirs(f"{self.cfg.result_dir}/timelapse", exist_ok=True)
        imageio.imwrite(
            f"{self.cfg.result_dir}/timelapse/{step:06d}.jpg", np.concatenate(tiles, axis=1)
        )
"""

HOOK = """
            if os.environ.get("TIMELAPSE_EVERY") and step % int(os.environ["TIMELAPSE_EVERY"]) == 0:
                self._timelapse(step)
"""


def main() -> None:
    path = Path(sys.argv[1])
    src = path.read_text()
    if "_timelapse" in src:
        print("already patched")
        return
    anchor = "    def train(self):"
    assert src.count(anchor) == 1, anchor
    src = src.replace(anchor, METHOD.lstrip("\n") + "\n" + anchor)
    hook_anchor = "            loss.backward()\n"
    assert src.count(hook_anchor) == 1, "loss.backward anchor"
    src = src.replace(hook_anchor, hook_anchor + HOOK.lstrip("\n"))
    path.write_text(src)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
