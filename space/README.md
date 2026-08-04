---
title: Residual Masking Network
emoji: 😀
colorFrom: purple
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: In-browser facial expression recognition (FER2013, ONNX)
models:
  - phamquiluan/ResidualMaskingNetwork
tags:
  - facial-expression-recognition
  - emotion-recognition
  - fer2013
  - onnx
  - webassembly
---

[ResMaskingNet](https://github.com/phamquiluan/ResidualMaskingNetwork) running
entirely in your browser. Images never leave your machine: YuNet locates faces
and ResMaskingNet classifies each crop into one of the seven FER2013 emotions,
all through onnxruntime-web on WebAssembly.

- Model weights: [phamquiluan/ResidualMaskingNetwork](https://huggingface.co/phamquiluan/ResidualMaskingNetwork)
- Paper: [Facial Expression Recognition using Residual Masking Network (ICPR 2020)](https://huggingface.co/papers/2603.05937)
- Python package: `pip install rmn`

## How it works

The 143M-parameter classifier is exported to ONNX and statically quantised to
int8 (526 MB to 132 MB) using FER2013 training images for calibration. On the
FER2013 private test split the quantised model scores within noise of the fp32
export, so the size reduction is effectively free.

The JavaScript reimplements the `rmn` preprocessing exactly rather than
approximating it: OpenCV's BGR-to-grayscale weights, the 1.1x square box
expansion from `convert_to_square`, OpenCV's bilinear pixel-center mapping, and
the plain `/255` scaling that `transforms.ToTensor()` applies. Verified against
the Python pipeline on the same image, agreement is 0px of box drift and under
0.002 absolute difference on every class probability.

Face detection letterboxes the input to 640x640 because the YuNet ONNX has a
fixed input shape; a plain square resize would stretch non-square photos and
shift the boxes.

## Performance

Roughly 0.9s per image single-threaded on a modern laptop. WASM multithreading
needs cross-origin isolation, which static Spaces do not provide, so the page
runs single-threaded and picks up threads automatically if those headers ever
appear.

Webcam mode analyses a frame every 2.5 seconds rather than every frame: at ~0.9s
per inference, per-frame analysis would queue up faster than it drains and the
page would fall behind. The video preview still repaints at display rate with
the latest boxes overlaid, so it looks live while the classifier runs on a
steady cadence. The period is measured start-to-start, and a run that overruns
it delays the next one instead of overlapping. Inference pauses while the tab is
hidden.

Activity is visible at all times: a dot pulses and the bar sweeps while a run is
in flight, the bar then drains toward the next scheduled run, and the status
line reports the frame number, face count, top emotion and per-frame time. Those
animations use `transform` and `opacity` so they stay on the compositor —
inference blocks the main thread for its duration, and a `width` or
`background-position` animation would freeze exactly when it needs to show
progress.

The 132 MB model downloads once and is stored in the Cache API, so repeat visits
skip the download.

## Citation

```bibtex
@inproceedings{pham2021facial,
  title={Facial expression recognition using residual masking network},
  author={Pham, Luan and Vu, The Huynh and Tran, Tuan Anh},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={4513--4519},
  year={2021},
  organization={IEEE}
}
```
