# Test fixtures

- `fox.glb` / `fox_rigged.fbx` - the Fox rigged/animated model (CC-BY 4.0,
  by PixelMannen, rigging/animation by @tomkranis, from the Khronos
  glTF-Sample-Models collection). `fox_rigged.fbx` is a Blender re-export of
  the same asset (`UpAxis = Z`, `UnitScaleFactor = 100`), used for
  FBX-vs-glTF skinning parity tests.
- `color_grid_uastc_zstd.ktx2` - a 1024x1024 Basis Universal UASTC texture with
  Zstandard supercompression, from the Filament project's `libs/ktxreader`
  tests (Apache-2.0). Real encoder output for the KTX2 transcode tests; the
  Draco and meshopt fixtures are encoded from `fox.glb` in the test itself.
