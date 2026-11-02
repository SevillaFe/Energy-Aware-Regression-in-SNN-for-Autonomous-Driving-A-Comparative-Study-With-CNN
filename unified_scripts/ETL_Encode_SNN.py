"""
ETL_Encode_SNN.py  (v1.1)

Encode the Udacity driving dataset into spike tensors with snnTorch
(rate, latency and delta) and track the encoding footprint with CodeCarbon.

Changes with respect to v1.0 (the version used for the published results
of P4 and of the thesis, kept unchanged as ETL_Encode_SNN_v1_published.py):

  * Delta encoding now computes a real inter-frame difference.
    snnTorch's ``spikegen.delta(data, threshold, padding, off_spike)`` takes
    a single tensor and differentiates it along its first (time) dimension.
    v1.0 passed the previous frame as a second positional argument, which
    snnTorch does not accept. v1.1 stacks S consecutive frames into a
    [S, C, H, W] tensor and calls ``delta`` on that tensor.
  * All frames of a delta window use the same camera (left, center or right),
    chosen once per sample, so that the difference reflects motion and not a
    change of viewpoint.
  * During training, all frames of a delta window receive the same random
    ColorJitter parameters, so that the difference does not reflect a change
    of augmentation.
  * Delta tensors are stored as (N, S, 1, 66, 200), the same layout as rate
    and latency. With ``padding=True`` the first time step is all zeros and
    steps 1..S-1 contain the spikes of each frame-to-frame change.

Rate and latency encodings are unchanged.
"""
import argparse
import os
import random

import h5py
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from codecarbon import EmissionsTracker
from PIL import Image
from snntorch.spikegen import delta, latency, rate
from torchvision import transforms
from tqdm import tqdm

IMAGE_SIZE = (66, 200)
STEERING_CORRECTION = 0.2
CAMERA_COLUMN = {"left": 0, "center": 1, "right": 2}
CAMERA_CORRECTION = {"left": STEERING_CORRECTION, "center": 0.0, "right": -STEERING_CORRECTION}
JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_h5_name(encoding, num_steps, gain, tau, threshold, delta_steps) -> str:
    if encoding == "rate":
        return f"encoded_dataset_rate_numsteps_{num_steps}_gain_{gain}.h5"
    if encoding == "latency":
        return f"encoded_dataset_latency_numsteps_{num_steps}_tau_{tau}.h5"
    if encoding == "delta":
        return f"encoded_dataset_delta_steps_{delta_steps}_threshold_{threshold}.h5"
    raise ValueError("Unsupported encoding type. Choose from 'rate', 'latency', or 'delta'.")


# ---------------------------------------------------------------- images
def camera_sample(row: pd.Series, img_dir: str, camera: str) -> tuple[str, float]:
    """Image path and steering label of one row for a given camera."""
    img_name = row.iloc[CAMERA_COLUMN[camera]].split("/")[-1]
    steering = float(row.iloc[3]) + CAMERA_CORRECTION[camera]
    return os.path.join(img_dir, img_name), steering


def sample_jitter():
    """Draw one set of ColorJitter parameters, to be applied to several frames."""
    cj = transforms.ColorJitter(**JITTER)
    return transforms.ColorJitter.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)


def apply_jitter(image: Image.Image, params) -> Image.Image:
    order, b, c, s, h = params
    for i in order:
        if i == 0 and b is not None:
            image = TF.adjust_brightness(image, b)
        elif i == 1 and c is not None:
            image = TF.adjust_contrast(image, c)
        elif i == 2 and s is not None:
            image = TF.adjust_saturation(image, s)
        elif i == 3 and h is not None:
            image = TF.adjust_hue(image, h)
    return image


def load_frame(path: str, normalize: transforms.Normalize, jitter_params=None) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = TF.resize(image, IMAGE_SIZE)
    if jitter_params is not None:
        image = apply_jitter(image, jitter_params)
    return normalize(TF.to_tensor(image))


# ---------------------------------------------------------------- HDF5
def create_datasets(h5_file, n_train: int, n_val: int, steps: int):
    shape = lambda n: (n, steps, 1, *IMAGE_SIZE)
    return (
        h5_file.create_dataset("train_images", shape=shape(n_train), dtype="float32"),
        h5_file.create_dataset("train_labels", shape=(n_train,), dtype="float32"),
        h5_file.create_dataset("val_images", shape=shape(n_val), dtype="float32"),
        h5_file.create_dataset("val_labels", shape=(n_val,), dtype="float32"),
    )


# ---------------------------------------------------------------- encoders
def process_split_static(split_data, encoding, img_dir, num_steps, gain, tau, use_augmentation, image_dset, label_dset):
    """Rate and latency: one frame per sample, encoded over num_steps (unchanged from v1.0)."""
    normalize = transforms.Normalize((0,), (1,))
    for idx in tqdm(range(len(split_data)), desc=f"{encoding} {'train' if use_augmentation else 'val'}"):
        camera = random.choice(list(CAMERA_COLUMN))
        path, steering = camera_sample(split_data.iloc[idx], img_dir, camera)
        frame = load_frame(path, normalize, sample_jitter() if use_augmentation else None)
        if encoding == "rate":
            encoded = rate(frame, num_steps=num_steps, gain=gain)
        else:
            encoded = latency(frame, num_steps=num_steps, tau=tau, normalize=True)
        image_dset[idx] = encoded.numpy()
        label_dset[idx] = steering


def process_split_delta(split_data, img_dir, delta_steps, threshold, off_spike, use_augmentation, image_dset, label_dset):
    """Delta: S consecutive frames, same camera and same augmentation, differentiated in time."""
    normalize = transforms.Normalize((0.5,), (0.5,))   # same pixel scale as v1.0, so thresholds keep their meaning
    first = delta_steps - 1
    for out_idx, idx in enumerate(tqdm(range(first, len(split_data)), desc=f"delta {'train' if use_augmentation else 'val'}")):
        camera = random.choice(list(CAMERA_COLUMN))
        jitter_params = sample_jitter() if use_augmentation else None
        frames, steering = [], None
        for j in range(idx - first, idx + 1):
            path, steering_j = camera_sample(split_data.iloc[j], img_dir, camera)
            frames.append(load_frame(path, normalize, jitter_params))
            steering = steering_j                       # label of the last (current) frame
        seq = torch.stack(frames)                       # [S, 1, 66, 200], oldest first
        encoded = delta(seq, threshold=threshold, padding=True, off_spike=off_spike)
        image_dset[out_idx] = encoded.numpy()           # step 0 is zeros; steps 1..S-1 are frame-to-frame changes
        label_dset[out_idx] = steering


def encode_and_save_to_h5(csv_file, img_dir, path_to_save, encoding="rate", num_steps=25, gain=0.5, tau=5.0,
                          threshold=0.5, delta_steps=2, off_spike=False, split_ratio=0.8) -> str:
    if encoding == "delta" and delta_steps < 2:
        raise ValueError("delta needs at least 2 consecutive frames (--delta-steps >= 2)")
    os.makedirs(path_to_save, exist_ok=True)
    h5_file_path = os.path.join(path_to_save, build_h5_name(encoding, num_steps, gain, tau, threshold, delta_steps))

    data = pd.read_csv(csv_file).reset_index(drop=True)   # rows must stay in capture order
    train_size = int(split_ratio * len(data))
    train_data = data.iloc[:train_size].reset_index(drop=True)
    val_data = data.iloc[train_size:].reset_index(drop=True)

    if encoding == "delta":
        steps = delta_steps
        n_train, n_val = max(0, len(train_data) - steps + 1), max(0, len(val_data) - steps + 1)
    else:
        steps = num_steps
        n_train, n_val = len(train_data), len(val_data)

    with h5py.File(h5_file_path, "w") as h5_file:
        tr_img, tr_lab, va_img, va_lab = create_datasets(h5_file, n_train, n_val, steps)
        for split, img_d, lab_d, aug in ((train_data, tr_img, tr_lab, True), (val_data, va_img, va_lab, False)):
            if encoding == "delta":
                process_split_delta(split, img_dir, delta_steps, threshold, off_spike, aug, img_d, lab_d)
            else:
                process_split_static(split, encoding, img_dir, num_steps, gain, tau, aug, img_d, lab_d)

    print(f"Encoded data saved to {h5_file_path}")
    return h5_file_path


def run_experiment(encoding, csv_file, img_dir, output_dir, emissions_dir, **params):
    os.makedirs(emissions_dir, exist_ok=True)
    name = "_".join([encoding] + [f"{k}_{v}" for k, v in params.items() if v is not None and k != "split_ratio"])
    tracker = EmissionsTracker(project_name=f"Experiment_{name}", output_dir=emissions_dir,
                               output_file=f"emissions_{name}.csv")
    tracker.start()
    try:
        encode_and_save_to_h5(csv_file, img_dir, output_dir, encoding=encoding, **params)
    finally:
        print(f"{name} emissions (kg CO2): {tracker.stop()}")


def parse_args():
    p = argparse.ArgumentParser(description="Encode driving dataset using snnTorch (rate, latency, delta), v1.1")
    p.add_argument("--data-dir", default="./Udacity_Dataset/Original_Images/")
    p.add_argument("--csv-file", default=None)
    p.add_argument("--img-dir", default=None)
    p.add_argument("--encoding", choices=["rate", "latency", "delta", "all"], default="all")
    p.add_argument("--num-steps", type=parse_int_list, default=[25, 50])
    p.add_argument("--gain", type=parse_float_list, default=[0.5, 1.0])
    p.add_argument("--tau", type=parse_float_list, default=[2.5, 5.0])
    p.add_argument("--threshold", type=parse_float_list, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--delta-steps", type=int, default=2, help="consecutive frames per delta sample (S >= 2)")
    p.add_argument("--off-spike", action="store_true")
    p.add_argument("--split-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="./Encoded_Images")
    p.add_argument("--emissions-dir", default="./Emissions")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    csv_file = args.csv_file or os.path.join(args.data_dir, "driving_log.csv")
    img_dir = args.img_dir or os.path.join(args.data_dir, "IMG")
    out = lambda enc: os.path.join(args.output_dir, enc.capitalize())
    emi = lambda enc: os.path.join(args.emissions_dir, enc.capitalize())

    if args.encoding in ("all", "rate"):
        for s in args.num_steps:
            for g in args.gain:
                run_experiment("rate", csv_file, img_dir, out("rate"), emi("rate"),
                               num_steps=s, gain=g, tau=None, threshold=None, split_ratio=args.split_ratio)
    if args.encoding in ("all", "latency"):
        for s in args.num_steps:
            for t in args.tau:
                run_experiment("latency", csv_file, img_dir, out("latency"), emi("latency"),
                               num_steps=s, gain=None, tau=t, threshold=None, split_ratio=args.split_ratio)
    if args.encoding in ("all", "delta"):
        for th in args.threshold:
            run_experiment("delta", csv_file, img_dir, out("delta"), emi("delta"),
                           threshold=th, delta_steps=args.delta_steps, off_spike=args.off_spike,
                           split_ratio=args.split_ratio)


if __name__ == "__main__":
    main()

# Example:
# python ETL_Encode_SNN.py --encoding delta --threshold 0.25,0.5 --delta-steps 2