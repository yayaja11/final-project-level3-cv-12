from os import scandir
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from module_fold import CEModule
import argparse
import os
from dataset import CustomDataset
import multiprocessing
from albumentations.augmentations.transforms import GaussNoise
import wandb


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def train(encoder, decoder, args):
    seed_everything(args.seed)
    run = wandb.init(project="Deep-drawing", entity="bcaitech_cv2")

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset, data loader -> 각 part 에 맞는 sketch를 잘라서 받아온다.
    train_dataset = CustomDataset(
        data_dir=args.sketch_dir, part=args.part, mode="train", transform=GaussNoise(var_limit=(0, 1), mean=0.5, per_channel=True, always_apply=False, p=0.5))
    val_dataset = CustomDataset(
        data_dir=args.sketch_dir, part=args.part, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count()//2,
                              shuffle=True,
                              pin_memory=use_cuda,
                              drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count()//2,
                            shuffle=True,
                            pin_memory=use_cuda,
                            drop_last=True)

    #--- Loss & optimizer & scheduler
    model = nn.Sequential(encoder, decoder)
    critetrion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=0.001, weight_decay=0.01)
    shcheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=20)

    columns = ["epoch", "mode", "input", "input with noise", "output"]
    test_table = wandb.Table(columns=columns)

    for epoch in range(argparse.epoch):
        model.train()
        loss_value = 0

        for step, (results, inputs) in enumerate(train_loader):
            inputs = inputs.unsqueeze(axis=1).float().to(device)
            results = results.unsqueeze(axis=1).float().to(device)
            model = model.to(device)
            outs = model(inputs)

            loss = critetrion(outs, results)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_table.add_data(epoch+1, "train", wandb.Image(results.squeeze(axis=1)[
                                0]), wandb.Image(inputs.squeeze(axis=1)[0]), wandb.Image(outs.squeeze(axis=1)[0]))

            loss_value += loss.item()
            wandb.log({"Train/loss": loss})
            if (step + 1) % 25 == 0:
                print(
                    f'Epoch [{epoch+1}/{args.epoch}], Step [{step+1}/{len(train_loader)}], Loss: {round(loss.item(),4)}')

        shcheduler.step()

        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            total_loss = 0.0
            cnt = 0

            for results, inputs in val_loader:
                inputs = inputs.unsqueeze(axis=1).float().to(device)
                results = results.unsqueeze(axis=1).float().to(device)
                outs = model(inputs)

                loss = critetrion(outs, results)
                total_loss += loss
                cnt += 1

                test_table.add_data(epoch+1, "val", wandb.Image(results.squeeze(axis=1)[0]), wandb.Image(
                    inputs.squeeze(axis=1)[0]), wandb.Image(outs.squeeze(axis=1)[0]))

            avrg_loss = total_loss / cnt
            print(
                f"Validation #{epoch} Average Loss : {round(avrg_loss.item(),4)}")
            wandb.log({"Val/Average loss": avrg_loss})
            save_model(encoder, saved_dir=args.save_dir,
                       file_name=f"encoder_{args.part}_latest")
            save_model(decoder, saved_dir=args.save_dir,
                       file_name=f"decoder_{args.part}_latest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, default='mouth',
                        help="Choose part name to encode")
    parser.add_argument('--seed', type=int, default=21, help="Fixing seed")
    parser.add_argument("--sketch_dir", type=str,
                        default="None", help="Loactaion of Sketch")
    parser.add_argument("--save_dir", type=str,
                        default="None", help="Loactaion to save pth")
    parser.add_argument("--epoch", type=int,
                        default=200, help="Number of epoch")
    parser.add_argument("--batch_size", type=int,
                        default=8, help="Size of batch")

    args = parser.parse_args()
    part = args.part

    part_encoder = CEModule.define_part_encoder(
        model=part, norm='instance', input_nc=1, latent_dim=512)
    part_decoder = CEModule.define_part_decoder(
        model=part, norm='instance', output_nc=1, latent_dim=512)

    train(part_encoder, part_decoder, args)
