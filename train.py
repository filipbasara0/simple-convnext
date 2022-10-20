import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torchinfo import summary

from datasets import load_dataset
from torchvision.transforms import (CenterCrop, Compose,
                                    ToTensor, RandAugment)
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing

from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from model import ConvNext
from utils import plot_losses, CustomLogger


def main(args):
    mixup_args = {
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'cutmix_minmax': None,
        'prob': 0.4,
        'switch_prob': 0.5,
        'mode': 'elem',
        'label_smoothing': 0.1,
        'num_classes': args.num_classes
    }

    mixup = Mixup(**mixup_args)
    rand_erasing = RandomErasing(probability=0.25, max_area=1/4, mode="pixel")

    # CIFAR10
    model = ConvNext(num_channels=3,
                     num_classes=args.num_classes,
                     patch_size=4,
                     layer_dims=[64, 128, 256, 512],
                     depths=[2, 2, 2, 2],
                     drop_rate=0.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    transforms_train = Compose([
        CenterCrop(args.resolution),
        RandAugment(num_ops=2),
        ToTensor()
    ])

    transforms_test = Compose([
        CenterCrop(args.resolution),
        ToTensor()
    ])

    if not args.dataset_name:
        raise ValueError(
            "You must specify a dataset name."
        )

    train_data, test_data = load_dataset(args.dataset_name,
                                            split=["train", "test"])

    def transforms_train_(examples):
        images = [
            transforms_train(image.convert("RGB"))
            for image in examples["img"]
        ]
        labels = [l for l in examples["label"]]
        return {"images": images, "labels": labels}

    def transforms_test_(examples):
        images = [
            transforms_test(image.convert("RGB"))
            for image in examples["img"]
        ]
        labels = [l for l in examples["label"]]
        return {"images": images, "labels": labels}

    train_data.set_transform(transforms_train_)
    test_data.set_transform(transforms_test_)

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                             max_lr=args.learning_rate,
                                             steps_per_epoch=len(train_dataloader),
                                             epochs=args.num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fnc = nn.CrossEntropyLoss()

    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
    logs_path = f"./training_logs/{current_date}/"
    os.makedirs(logs_path, exist_ok=True)
    logger = CustomLogger("simple-convnext",
                            file_path=f"{logs_path}/training_log.txt")
    model_summary = str(summary(model, (1, 3, args.resolution, args.resolution),  verbose=0))
    logger.log_info(model_summary)

    global_step = 0
    losses = []
    valid_losses = []
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0
        for step, batch in enumerate(train_dataloader):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            images, labels = mixup(images, labels)
            images = rand_erasing(images)

            preds = model(images)

            loss = loss_fnc(preds, labels)
            loss.backward()

            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            losses_log += loss.detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }

            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        losses.append(losses_log / (step + 1))
        if epoch % args.save_model_epochs == 0:
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                valid_labels = []
                valid_preds = []
                for step, batch in enumerate(
                        tqdm(test_dataloader, total=len(test_dataloader))):
                    images = batch["images"].to(device)
                    labels = batch["labels"].to(device)

                    preds = model(images)

                    loss = loss_fnc(preds, labels)
                    valid_loss += loss.item()

                    preds = preds.argmax(dim=-1)
                    valid_labels.extend(labels.detach().cpu().tolist())
                    valid_preds.extend(preds.detach().cpu().tolist())

                # print for debug
                print(f"Valid loss: {valid_loss / len(test_dataloader)}")
                print(classification_report(valid_labels, valid_preds))
                
                logger.log_info(f"Epoch {epoch}")
                logger.log_info(logs)
                logger.log_info(
                    f"Valid loss: {valid_loss / len(test_dataloader)}")
                logger.log_info(classification_report(valid_labels,
                                                      valid_preds))

                torch.save(
                    {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, args.output_dir)

            epoch_path = f"{logs_path}/{epoch}"
            os.makedirs(epoch_path)

            valid_losses.append(valid_loss / len(test_dataloader))
            plot_losses(train_losses=losses,
                        valid_losses=valid_losses,
                        path=epoch_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir",
                        type=str,
                        default=None,
                        help="A folder containing the training data.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="trained_models/cifar10.pth")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=5e-2) #1e-1
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")

    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)