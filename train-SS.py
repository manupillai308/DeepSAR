from modules.dataloader import XView3Data
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import json
import os
from tqdm.auto import tqdm
from modules.featextract import FeatureExtractor
from modules.model import RPN
from modules.config import load_model_config
from torchvision.models import resnet50
from torchvision import io, transforms as T
import argparse

def preprocess_label(df):
    df = df.dropna(subset=["is_vessel"])
    return df


def loss_fn(binary_pred, cls_true):

    losses = torch.nn.functional.binary_cross_entropy(binary_pred, cls_true, reduction='none')
    label_arr = cls_true.squeeze(1).detach().cpu().numpy()

    b, bg_i, bg_j = np.where(label_arr == 0)
    b, fg_i, fg_j = np.where(label_arr == 1)

    fg_l = len(fg_i)
    bg_l = len(bg_i)
    ixs = np.random.choice(np.arange(bg_l), fg_l)

    fg_loss = losses[b, :, fg_i, fg_j].mean()
    bg_loss = losses[b, :, bg_i[ixs], bg_j[ixs]].mean()

    return fg_loss + bg_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', required=True, help="Path to the directory where the model checkpoints and tensorboard logs are to be saved.")
    parser.add_argument('--split-file', help="Path to json file with train-test split ids. (Optional)")
    parser.add_argument('-v', default=False, type=bool, help="Verbose for dataloading")
    parser.add_argument('--data-path', required=True, help="Path to the training data directory.")
    parser.add_argument('--label-path', required=True, help="Path to the training label directory.")
    args = parser.parse_args()

        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



    transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    result_path = args.result_dir
    split_file = args.split_file
    cv_split = {"full":{"train":[], "test":[]}}
    if split_file:
        cv_split = json.load(open(split_file))

    outer_pbar = tqdm(cv_split.keys(), position=0)

    for cv in outer_pbar:
        outer_pbar.set_description(f"Running split {cv}")
        os.makedirs(os.path.join(result_path, cv), exist_ok=True)
        cv_path = os.path.join(result_path, cv)
                        
        ignore_id =  cv_split[cv]["test"]
        
        backbone = resnet50(pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = False
        backbone = backbone.eval()

        config = load_model_config(backbone)



        feat = FeatureExtractor(config)
        rpn = RPN(128)
        model = torch.nn.Sequential(*[feat, rpn])

        train_data_path = args.data_path
        train_label_path = args.label_path

        train_data = XView3Data(ignore_id=ignore_id, background_chip_ratio=0.1, obj_size=5, threshold=0.25, overwrite=False,
                                labels_path=train_label_path, data_path=train_data_path, preprocess_label=preprocess_label, shore=False, verbose=args.v)

        batch_size = 8
        n_epochs = 5

        train_sampler = torch.utils.data.RandomSampler(train_data)


        data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)

        runs_path = os.path.join(cv_path, "runs")
        os.makedirs(runs_path, exist_ok=True)
        writer = SummaryWriter(os.path.join(runs_path, "factseg_experiment_RPN"))

        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        running_loss = 0

        for epoch in range(n_epochs):
            num_nans = 0
            inner_pbar = tqdm(total=len(data_loader_train), position=1, leave=False, ascii=True, desc=f"Epoch: {epoch+1} Step")
            for i, data in enumerate(data_loader_train, 1):

                un_img, class_labels, inst_weight = data
                img = transform(un_img)
                img = img.to(device)

                class_labels = class_labels.cpu().numpy()
                class_labels[class_labels != 0] = 1
                class_labels = torch.from_numpy(class_labels.astype("float32")).unsqueeze(1).to(device)


                optimizer.zero_grad()

                pred = model(img)

                loss = loss_fn(pred, class_labels)

                inner_pbar.update(1)

                if torch.isnan(loss):
                    if num_nans > 10:
                        raise RuntimeError(f"Model Error: Encountered {num_nans} nan loss")
                    num_nans += 1
                    continue
                loss.backward()

                optimizer.step()

                running_loss += loss.detach().item()

                num_nans = 0
                if i%15 == 0:

                    writer.add_scalar('training_loss', running_loss/15, epoch*len(data_loader_train)+i)
                    running_loss = 0.0

            scheduler.step()
            
            ckpt_path = os.path.join(cv_path, "ckpts")
            os.makedirs(ckpt_path, exist_ok=True)
            
            checkpoint_pth = os.path.join(ckpt_path, f'RPN_trained_model_{epoch+1}_epochs.pth')
            torch.save(model.state_dict(), checkpoint_pth)
            inner_pbar.close()

        writer.close()





