import argparse
import os
import random

import h5py
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from PIL import Image
from snntorch.spikegen import delta, latency, rate
from tqdm import tqdm
from torchvision import transforms


IMAGE_SIZE = (66, 200)
STEERING_CORRECTION = 0.2


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_h5_name(
    encoding: str,
    num_steps: int | None,
    gain: float | None,
    tau: float | None,
    threshold: float | None,
) -> str:
    if encoding == "rate":
        return f"encoded_dataset_rate_numsteps_{num_steps}_gain_{gain}.h5"
    if encoding == "latency":
        return f"encoded_dataset_latency_numsteps_{num_steps}_tau_{tau}.h5"
    if encoding == "delta":
        return f"encoded_dataset_delta_threshold_{threshold}.h5"
    raise ValueError("Unsupported encoding type. Choose from 'rate', 'latency', or 'delta'.")


def choose_camera_sample(row: pd.Series, img_dir: str, steering_correction: float) -> tuple[str, float]:
    camera_choice = random.choice(["left", "center", "right"])
    if camera_choice == "left":
        img_name = row.iloc[0].split("/")[-1]
        steering = row.iloc[3] + steering_correction
    elif camera_choice == "right":
        img_name = row.iloc[2].split("/")[-1]
        steering = row.iloc[3] - steering_correction
    else:
        img_name = row.iloc[1].split("/")[-1]
        steering = row.iloc[3]
    return os.path.join(img_dir, img_name), float(steering)


def preprocess_image(
    image_path: str,
    base_transform: transforms.Resize,
    to_tensor: transforms.ToTensor,
    normalize: transforms.Normalize,
    augment_transform: transforms.Compose | None,
) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    image = base_transform(image)
    if augment_transform is not None:
        image = augment_transform(image)
    image = to_tensor(image)
    image = normalize(image)
    return image


def encode_tensor(
    encoding: str,
    image: torch.Tensor,
    prev_image: torch.Tensor | None,
    num_steps: int | None,
    gain: float | None,
    tau: float | None,
    threshold: float | None,
    padding: bool,
    off_spike: bool,
) -> torch.Tensor:
    if encoding == "rate":
        if num_steps is None or gain is None:
            raise ValueError("rate encoding requires num_steps and gain")
        return rate(image, num_steps=num_steps, gain=gain)

    if encoding == "latency":
        if num_steps is None or tau is None:
            raise ValueError("latency encoding requires num_steps and tau")
        return latency(image, num_steps=num_steps, tau=tau, normalize=True)

    if encoding == "delta":
        if prev_image is None or threshold is None:
            raise ValueError("delta encoding requires prev_image and threshold")
        return delta(image, prev_image, threshold=threshold, padding=padding, off_spike=off_spike)

    raise ValueError("Unsupported encoding type. Choose from 'rate', 'latency', or 'delta'.")


def create_datasets(
    h5_file: h5py.File,
    encoding: str,
    num_train_samples: int,
    num_val_samples: int,
    num_steps: int | None,
):
    if encoding == "delta":
        train_img_dset = h5_file.create_dataset("train_images", shape=(num_train_samples, 1, 66, 200), dtype="float32")
        train_label_dset = h5_file.create_dataset("train_labels", shape=(num_train_samples,), dtype="float32")
        val_img_dset = h5_file.create_dataset("val_images", shape=(num_val_samples, 1, 66, 200), dtype="float32")
        val_label_dset = h5_file.create_dataset("val_labels", shape=(num_val_samples,), dtype="float32")
    else:
        if num_steps is None:
            raise ValueError("num_steps is required for rate and latency encodings")
        train_img_dset = h5_file.create_dataset(
            "train_images", shape=(num_train_samples, num_steps, 1, 66, 200), dtype="float32"
        )
        train_label_dset = h5_file.create_dataset("train_labels", shape=(num_train_samples,), dtype="float32")
        val_img_dset = h5_file.create_dataset(
            "val_images", shape=(num_val_samples, num_steps, 1, 66, 200), dtype="float32"
        )
        val_label_dset = h5_file.create_dataset("val_labels", shape=(num_val_samples,), dtype="float32")

    return train_img_dset, train_label_dset, val_img_dset, val_label_dset


def process_split(
    split_data: pd.DataFrame,
    encoding: str,
    img_dir: str,
    num_steps: int | None,
    gain: float | None,
    tau: float | None,
    threshold: float | None,
    padding: bool,
    off_spike: bool,
    use_augmentation: bool,
    image_dset,
    label_dset,
):
    base_transform = transforms.Resize(IMAGE_SIZE)
    augment_transform = transforms.Compose(
        [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]
    )
    to_tensor = transforms.ToTensor()

    if encoding == "delta":
        normalize = transforms.Normalize((0.5,), (0.5,))
        iterator = range(1, len(split_data))
    else:
        normalize = transforms.Normalize((0,), (1,))
        iterator = range(len(split_data))

    progress = tqdm(iterator, total=len(iterator), desc=f"Processing {encoding} {'train' if use_augmentation else 'val'}")

    for out_idx, idx in enumerate(progress):
        row = split_data.iloc[idx]
        img_path, steering = choose_camera_sample(row, img_dir, STEERING_CORRECTION)

        aug = augment_transform if use_augmentation else None
        image_tensor = preprocess_image(img_path, base_transform, to_tensor, normalize, aug)

        prev_tensor = None
        if encoding == "delta":
            prev_row = split_data.iloc[idx - 1]
            prev_img_path, _ = choose_camera_sample(prev_row, img_dir, STEERING_CORRECTION)
            prev_tensor = preprocess_image(prev_img_path, base_transform, to_tensor, normalize, aug)

        encoded = encode_tensor(
            encoding=encoding,
            image=image_tensor,
            prev_image=prev_tensor,
            num_steps=num_steps,
            gain=gain,
            tau=tau,
            threshold=threshold,
            padding=padding,
            off_spike=off_spike,
        ).numpy()

        image_dset[out_idx] = encoded
        label_dset[out_idx] = steering
def encode_and_save_to_h5(
    csv_file: str,
    img_dir: str,
    path_to_save: str,
    encoding: str = "rate",
    num_steps: int | None = 30,
    gain: float | None = 1.0,
    tau: float | None = 1.0,
    threshold: float | None = 0.5,
    padding: bool = False,
    off_spike: bool = False,
    split_ratio: float = 0.8,
) -> str:
    os.makedirs(path_to_save, exist_ok=True)

    h5_filename = build_h5_name(encoding, num_steps, gain, tau, threshold)
    h5_file_path = os.path.join(path_to_save, h5_filename)

    data = pd.read_csv(csv_file).reset_index(drop=True)
    train_size = int(split_ratio * len(data))
    train_data = data.iloc[:train_size].reset_index(drop=True)
    val_data = data.iloc[train_size:].reset_index(drop=True)

    if encoding == "delta":
        num_train_samples = max(0, len(train_data) - 1)
        num_val_samples = max(0, len(val_data) - 1)
    else:
        num_train_samples = len(train_data)
        num_val_samples = len(val_data)

    with h5py.File(h5_file_path, "w") as h5_file:
        train_img_dset, train_label_dset, val_img_dset, val_label_dset = create_datasets(
            h5_file=h5_file,
            encoding=encoding,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            num_steps=num_steps,
        )

        process_split(
            split_data=train_data,
            encoding=encoding,
            img_dir=img_dir,
            num_steps=num_steps,
            gain=gain,
            tau=tau,
            threshold=threshold,
            padding=padding,
            off_spike=off_spike,
            use_augmentation=True,
            image_dset=train_img_dset,
            label_dset=train_label_dset,
        )

        process_split(
            split_data=val_data,
            encoding=encoding,
            img_dir=img_dir,
            num_steps=num_steps,
            gain=gain,
            tau=tau,
            threshold=threshold,
            padding=padding,
            off_spike=off_spike,
            use_augmentation=False,
            image_dset=val_img_dset,
            label_dset=val_label_dset,
        )

    print(f"Encoded data saved to {h5_file_path}")
    return h5_file_path


def run_experiment(
    encoding: str,
    csv_file: str,
    img_dir: str,
    output_dir: str,
    emissions_dir: str,
    num_steps: int | None,
    gain: float | None,
    tau: float | None,
    threshold: float | None,
    split_ratio: float,
    padding: bool,
    off_spike: bool,
):
    os.makedirs(emissions_dir, exist_ok=True)
    name_parts = [encoding]
    if num_steps is not None:
        name_parts.append(f"steps_{num_steps}")
    if gain is not None:
        name_parts.append(f"gain_{gain}")
    if tau is not None:
        name_parts.append(f"tau_{tau}")
    if threshold is not None:
        name_parts.append(f"threshold_{threshold}")
    experiment_name = "_".join(name_parts)

    tracker = EmissionsTracker(
        project_name=f"Experiment_{experiment_name}",
        output_dir=emissions_dir,
        output_file=f"emissions_{experiment_name}.csv",
    )

    tracker.start()
    try:
        encode_and_save_to_h5(
            csv_file=csv_file,
            img_dir=img_dir,
            path_to_save=output_dir,
            encoding=encoding,
            num_steps=num_steps,
            gain=gain,
            tau=tau,
            threshold=threshold,
            split_ratio=split_ratio,
            padding=padding,
            off_spike=off_spike,
        )
    finally:
        emissions = tracker.stop()
        print(f"{experiment_name} emissions (kg CO2): {emissions}")


def parse_args():
    parser = argparse.ArgumentParser(description="Encode driving dataset using snnTorch (rate, latency, delta)")

    parser.add_argument("--data-dir", default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images/")
    parser.add_argument("--csv-file", default=None)
    parser.add_argument("--img-dir", default=None)

    parser.add_argument("--encoding", choices=["rate", "latency", "delta", "all"], default="all")
    parser.add_argument("--num-steps", type=parse_int_list, default=[25, 50])
    parser.add_argument("--gain", type=parse_float_list, default=[0.5, 1.0])
    parser.add_argument("--tau", type=parse_float_list, default=[2.5, 5.0])
    parser.add_argument("--threshold", type=parse_float_list, default=[0.25, 0.5, 0.75, 1.0])

    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--padding", action="store_true")
    parser.add_argument("--off-spike", action="store_true")

    parser.add_argument(
        "--output-rate",
        default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Encoded_Datasets",
    )
    parser.add_argument(
        "--output-latency",
        default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Encoded_Datasets",
    )
    parser.add_argument(
        "--output-delta",
        default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta/Encoded_Datasets",
    )

    parser.add_argument(
        "--emissions-rate",
        default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Emissions",
    )
    parser.add_argument(
        "--emissions-latency",
        default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Emissions",
    )
    parser.add_argument(
        "--emissions-delta",
        default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta/Emissions",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    csv_file = args.csv_file or os.path.join(args.data_dir, "driving_log.csv")
    img_dir = args.img_dir or os.path.join(args.data_dir, "IMG")

    run_rate = args.encoding in ("all", "rate")
    run_latency = args.encoding in ("all", "latency")
    run_delta = args.encoding in ("all", "delta")

    if run_rate:
        for num_steps in args.num_steps:
            for gain in args.gain:
                run_experiment(
                    encoding="rate",
                    csv_file=csv_file,
                    img_dir=img_dir,
                    output_dir=args.output_rate,
                    emissions_dir=args.emissions_rate,
                    num_steps=num_steps,
                    gain=gain,
                    tau=None,
                    threshold=None,
                    split_ratio=args.split_ratio,
                    padding=args.padding,
                    off_spike=args.off_spike,
                )

    if run_latency:
        for num_steps in args.num_steps:
            for tau in args.tau:
                run_experiment(
                    encoding="latency",
                    csv_file=csv_file,
                    img_dir=img_dir,
                    output_dir=args.output_latency,
                    emissions_dir=args.emissions_latency,
                    num_steps=num_steps,
                    gain=None,
                    tau=tau,
                    threshold=None,
                    split_ratio=args.split_ratio,
                    padding=args.padding,
                    off_spike=args.off_spike,
                )

    if run_delta:
        for threshold in args.threshold:
            run_experiment(
                encoding="delta",
                csv_file=csv_file,
                img_dir=img_dir,
                output_dir=args.output_delta,
                emissions_dir=args.emissions_delta,
                num_steps=None,
                gain=None,
                tau=None,
                threshold=threshold,
                split_ratio=args.split_ratio,
                padding=args.padding,
                off_spike=args.off_spike,
            )


if __name__ == "__main__":
    main()
    
# Ejemplo rápido de ejecución:
# python 00_ETL_Encode_SNN_delta.py --encoding delta --threshold 0.25,0.5 --split-ratio 0.8


