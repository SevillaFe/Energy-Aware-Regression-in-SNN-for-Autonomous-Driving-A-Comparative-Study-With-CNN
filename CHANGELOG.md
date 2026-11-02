# Changelog

## v1.1

`ETL_Encode_SNN.py` now implements delta encoding as an inter-frame difference over consecutive frames.

**What changed in delta encoding**

- **Call to snnTorch.** `snntorch.spikegen.delta(data, threshold, padding, off_spike)` takes one tensor and differentiates it along its first (time) dimension. v1.0 passed the previous frame as a second positional argument, which does not match that signature (checked with snnTorch 0.7.0 and 1.0.0). v1.1 stacks S consecutive frames into a `[S, 1, 66, 200]` tensor and calls `delta` on it.
- **Same camera in each window.** The camera (left, center or right) is chosen once per sample and used for every frame in the window. The spikes therefore reflect motion between frames, not a change of viewpoint.
- **Same augmentation in each window.** During training, every frame in a window gets the same random ColorJitter parameters.
- **Output shape.** Delta datasets are stored as `(N, S, 1, 66, 200)`, the same layout as rate and latency. Step 0 is all zeros (`padding=True`). Steps 1 to S-1 hold the spikes for each frame-to-frame change.
- **New option.** `--delta-steps S` sets the number of consecutive frames (default 2, minimum 2).
- **Unchanged pixel scale.** Pixel values are still normalized to [-1, 1], so frame differences lie in [-2, 2] and the `--threshold` values keep the meaning they had in v1.0.

**Other changes**

- Added `--seed` for reproducible camera choice and augmentation.
- Rate and latency encoding behave as in v1.0.

**Expected impact on results.** v1.1 can produce delta datasets, and models trained on them, that differ from those reported for delta encoding in the published article. Rate and latency results are not affected.

## v1.0-published

This is the script as used for the article and kept unchanged for traceability (tag `v1.0-published`, file `ETL_Encode_SNN_v1_published.py`).
