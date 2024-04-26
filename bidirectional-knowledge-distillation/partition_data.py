import argparse
import os
import pickle

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100

from modules.data_utils import TinyImageNet200


def parse_args():
    parser = argparse.ArgumentParser(description="Partition data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "tiny-imagenet-200"],
    )
    parser.add_argument("--cloud-ratio", type=float, default=0.1)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()

    # Print arguments
    print(f"Running with arguments: {args}")

    if args.dataset.startswith("cifar"):
        transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    if args.dataset == "cifar10":
        train_data = CIFAR10(root=args.data_dir, train=True, transform=transform)
    elif args.dataset == "cifar100":
        train_data = CIFAR100(root=args.data_dir, train=True, transform=transform)
    else:
        train_data = TinyImageNet200(
            root=os.path.join(args.data_dir),
            split="train",
            transform=transform,
        )

    num_total = len(train_data)
    num_cloud = int(args.cloud_ratio * num_total)
    num_device = num_total - num_cloud

    train_data_cloud, train_data_device = random_split(
        train_data, [num_cloud, num_device]
    )

    # Split the device data into `num_splits` splits
    split_sizes = [num_device // args.num_splits] * args.num_splits
    split_sizes[-1] += num_device % args.num_splits
    train_data_device_splits = random_split(train_data_device, split_sizes)

    # Save the partitioned data
    with open(os.path.join(args.data_dir, f"pkl/{args.dataset}_cloud.pkl"), "wb") as f:
        pickle.dump(train_data_cloud, f)

    for i, subset in enumerate(train_data_device_splits):
        with open(
            os.path.join(args.data_dir, f"pkl/{args.dataset}_device_{i}.pkl"), "wb"
        ) as f:
            pickle.dump(subset, f)


if __name__ == "__main__":
    main()
