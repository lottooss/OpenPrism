# Recorded Aimlabs test

[Play the 53-second recording](demo.mp4) · [Final scorecard](result.jpg) · [Selected runtime metrics](demo-results.json)

Recorded on 13 September 2026 in the controlled Gridshot Ultimate remake task, using the local native C++ runtime and a TensorRT reference detector. The runtime was armed for 45 seconds within the 60-second round. Aimlabs reported **16 hits from 16 shots, 100% accuracy and a score of 3,024**. This is a small single-run sample, not an accuracy guarantee.

The clip is a continuous normal-speed segment. Only the waiting period before the countdown was trimmed; no aiming sequences were removed or sped up. The video ends before the score screen; the separate scorecard was captured after the same round. H.264 video is 1280×720 at 30 fps, with no audio. Capture and software encoding ran concurrently, so this is not a clean performance benchmark.

Environment: Windows x64, RTX 4060 Laptop GPU, 1920×1080 game output, FOV 103, sensitivity 1, TensorRT 10.8 and CUDA 12.6. The model was built locally and is not included in this source distribution. Current acquisition is visibly slow; the run demonstrates the closed loop and conservative firing, not competitive speed.

The in-game task author and game interface are third-party content. OpenPrism is not affiliated with Aimlabs.
