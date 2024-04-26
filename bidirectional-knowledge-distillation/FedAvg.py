import argparse
import os
import pickle

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import mobilenet_v3_small

from modules.data_utils import TinyImageNet200
from modules.functional import fed_avg, test_accuracy, train


def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "tiny-imagenet-200"],
    )
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--model", type=str, default="mobilenet_v3_small")
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-save-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    # Print the arguments
    print(f"Running with arguments: {args}")

    train_data_splits = []
    for i in range(args.num_splits):
        data_path = os.path.join(args.data_dir, f"pkl/{args.dataset}_device_{i}.pkl")
        with open(data_path, "rb") as f:
            train_data_splits.append(pickle.load(f))

    val_data_ratio = 0.1
    val_data_splits = []

    for i, subset in enumerate(train_data_splits):
        val_size = int(val_data_ratio * len(subset))
        train_size = len(subset) - val_size
        train_subset, val_subset = random_split(subset, [train_size, val_size])
        train_data_splits[i] = train_subset
        val_data_splits.append(val_subset)

    if args.dataset == "cifar10":
        test_data = CIFAR10(
            root=args.data_dir,
            train=False,
            transform=T.Compose(
                [
                    T.ToImage(),
                    T.ToDtype(torch.float32),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            ),
        )
        num_classes = 10
    elif args.dataset == "cifar100":
        test_data = CIFAR100(
            root=args.data_dir,
            train=False,
            transform=T.Compose(
                [
                    T.ToImage(),
                    T.ToDtype(torch.float32),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            ),
        )
        num_classes = 100
    else:
        test_data = TinyImageNet200(
            root=args.data_dir,
            split="val",
            transform=T.Compose(
                [
                    T.ToImage(),
                    T.ToDtype(torch.float32),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        num_classes = 200

    if args.model == "mobilenet_v3_small":
        model_global = mobilenet_v3_small(weights=None, num_classes=num_classes)

    accuracy = test_accuracy(model_global, test_data)
    print(f"Round 0: {accuracy}")

    accuracy_list = []
    accuracy_list.append(accuracy)

    best_accuracy = 0.0

    for round_num in range(args.num_rounds):
        model_weights = []
        for i in range(args.num_splits):
            if args.model == "mobilenet_v3_small":
                model_device = mobilenet_v3_small(weights=None, num_classes=num_classes)

            model_device.load_state_dict(model_global.state_dict())
            train(
                model_device,
                train_data_splits[i],
                val_data_splits[i],
                checkpoint_save_path=os.path.join(
                    args.checkpoint_save_dir,
                    f"{args.model}_{args.dataset}_device_{i}.pth",
                ),
            )
            model_weights.append(model_device.state_dict())

        model_global.load_state_dict(
            fed_avg(
                model_weights, [len(train_data) for train_data in train_data_splits]
            )
        )

        accuracy = test_accuracy(model_global, test_data)
        accuracy_list.append(accuracy)
        print(f"Round {round_num+1}: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                model_global.state_dict(),
                os.path.join(
                    args.checkpoint_save_dir,
                    f"{args.model}_{args.dataset}_aggregated.pth",
                ),
            )

    # Print results
    for i, accuracy in enumerate(accuracy_list):
        print(f"Round {i}: {accuracy}")


if __name__ == "__main__":
    main()
