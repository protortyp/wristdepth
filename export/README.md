# Export

The ONNX graphs and TensorRT engine come from
[depthpro2-onnx](https://github.com/protortyp/depthpro2-onnx) at `1afa858`, with
[`depthpro2-onnx-weights.patch`](depthpro2-onnx-weights.patch) adding a `--weights` option
(fine-tuned state dict) to its export, fixture and parity scripts.

```sh
git clone https://github.com/protortyp/depthpro2-onnx && cd depthpro2-onnx
git checkout 1afa858 && git apply ../wristdepth/export/depthpro2-onnx-weights.patch
uv run --extra export python fixtures/make_fixture.py --size B --weights wristdepth.pt --name WristDepth-B --clips 4 --windows 4
uv run --extra export python export_onnx.py --size B --weights wristdepth.pt --out out/WristDepth-B --clip 4 --stream 4
uv run python verify_parity.py out/WristDepth-B/onnx/model_fp16.onnx --refs WristDepth-B
```

`wristdepth.pt` is a state dict, e.g. `DepthPro2.from_pretrained("christianengel/WristDepth").state_dict()`
saved with `torch.save`.
