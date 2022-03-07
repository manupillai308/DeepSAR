from email.policy import default
from modules.dataloader import XView3DDN
import numpy as np
from tqdm.auto import tqdm
import os
import json
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from torchvision import io, transforms as T
from modules.featextract import FeatureExtractor
from modules.model import RPN, DN
from modules.config import load_model_config
from modules.utils import evaluate, save_fig, convert_prob_to_image
from modules.losses import loss_fn_dn as loss_dn1
from modules.losses import loss_fn_dn_2 as loss_dn2
import argparse

def prepare_backbone():
    backbone = resnet50(pretrained=True)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.eval()
    
    return backbone


def preprocess_label(df):
    df = df.dropna(subset=["is_vessel"])
    return df

def rpn_callable_and_loss(rpn, decoy):
    def func(x):
        x = transform(x)
        if decoy:
            return torch.ones_like(x), loss_dn2
        else:
            return rpn(x.to(device_rpn)).to(torch.device
        ("cpu")), loss_dn1
    return func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', required=True, help="Path to the directory where the result checkpoints and tensorboard logs are to be saved.")
    parser.add_argument('--split-file', help="Path to json file with train-test split ids. (Optional)")
    parser.add_argument('-v', default=False, type=bool, help="Verbose for dataloading")
    parser.add_argument('--data-path', required=True, help="Path to the training data directory.")
    parser.add_argument('--label-path', required=True, help="Path to the training label directory.")
    parser.add_argument('--finetune', type=bool, required=True, help="Finetune the model")
    parser.add_argument('--rpn-path', default=None, help="Path to trained RPN model ckpt")
    parser.add_argument('--dn-path', default=None, help="Path to trained DN model ckpt if finetuning")
    args = parser.parse_args()

    device_rpn = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_dn = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



    transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    finetune = args.finetune
    if finetune:
        assert args.dn_path, f"--dn-path must be provided for finetune {finetune}"
    else:
        assert args.rpn_path, f"--rpn-path must be provided for finetune {finetune}"
        
    result_path = args.result_dir
    split_file = args.split_file
    cv_split = {"full":{"train":[], "test":[]}}
    if split_file:
        cv_split = json.load(open("cv_split.json"))

    rpn_name = args.rpn_path
    dn_name = args.dn_path

    outer_pbar = tqdm(cv_split.keys(), position=0)

    if finetune:
        lr = 0.0001
    else:
        lr = 0.001
    for cv in outer_pbar:
        outer_pbar.set_description(f"Running split {cv}")
        os.makedirs(os.path.join(result_path, cv), exist_ok=True)
        cv_path = os.path.join(result_path, cv)
                        
        ignore_id =  cv_split[cv]["test"]
        model_path = os.path.join(cv_path, "ckpts")
        rpn = None
        if not finetune:
            rpn = torch.nn.Sequential(*[FeatureExtractor(load_model_config(prepare_backbone())), RPN(128)])
            rpn.load_state_dict(torch.load(os.path.join (model_path, rpn_name)))

            for param in rpn.parameters():
                param.requires_grad = False

            rpn.eval()
            rpn.to(device_rpn)
            
        rpn_fn, loss_fn = rpn_callable_and_loss(rpn, finetune)
        model = torch.nn.Sequential(*[FeatureExtractor(load_model_config(prepare_backbone())), DN(128, 4)])
        if finetune:
            model.load_state_dict(torch.load(os.path.join(model_path, dn_name)))

        train_data_path = args.data_path
        train_label_path = args.label_path

        train_data = XView3DDN(ignore_id=ignore_id, model_callable=rpn_fn, background_chip_ratio=0.1, obj_size=5, threshold=0.25, overwrite=False, labels_path=train_label_path, data_path=train_data_path, preprocess_label=preprocess_label, shore=False, verbose=args.v)


        batch_size = 8
        n_epochs = 3

        train_sampler = torch.utils.data.RandomSampler(train_data)


        data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)


        runs_path = os.path.join(cv_path, "runs")
        os.makedirs(runs_path, exist_ok=True)
        writer = SummaryWriter(os.path.join(runs_path, f"factseg_experiment_DN{1+finetune}"))

    #     model.train()
        model.to(device_dn)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



        running_loss = 0

        for epoch in range(n_epochs):
            num_nans = 0
            inner_pbar = tqdm(total=len(data_loader_train), position=1, leave=False, ascii=True, desc=f"Epoch: {epoch+1} Step")
            for i, data in enumerate(data_loader_train, 1):

                proposal_t, un_img, class_labels, inst_weight = data
                img = transform(un_img)
                img = img.to(device_dn)

                class_labels = class_labels.to(device_dn)
                optimizer.zero_grad()

                pred = model(img)

                if finetune:
                    loss = loss_fn(pred, class_labels)
                else:
                    loss = loss_fn(proposal_t, pred, class_labels)
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

                    step = None

                    running_loss = 0.0

            scheduler.step()
            ckpt_path = os.path.join(cv_path, "ckpts")
            os.makedirs(ckpt_path, exist_ok=True)
            
            checkpoint_pth = os.path.join(ckpt_path, f'DN{1+finetune}_trained_model_{epoch+1}_epochs.pth')
            torch.save(model.state_dict(), checkpoint_pth)
            inner_pbar.close()

        writer.close()





