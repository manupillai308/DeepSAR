from modules.dataloader import XView3Data
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from modules.metric import score, compute_loc_performance
import numpy as np
from tqdm import tqdm
import cv2, os
import rasterio
from rasterio.enums import Resampling
import torchvision
from torch.utils.tensorboard import SummaryWriter
import joblib
import json
from torchvision.models import resnet50
from torchvision import io, transforms as T
from modules.featextract import FeatureExtractor
from modules.model import RPN, DN
from modules.config import load_model_config
from modules.utils import evaluate, save_fig, convert_prob_to_image, prepare_label
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


def get_detections(joint_prob):

    detections = []
    y_arr = joint_prob.detach().cpu().numpy()

    y_cls = np.argmax(y_arr, axis=0)
    y_prob = y_arr.max(axis=0)
    y_prob_fg = 1 - y_arr[0, :, :]

    y_bin = ((y_prob_fg >= 0.5)*(y_cls!=0)).astype(np.uint8)
    contours, _ = cv2.findContours(y_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        ix, iy = contour[:, :, 0], contour[:, :, 1]
        M = cv2.moments(contour)
        if M['m00'] == 0 and len(contour) != 0:
            cx = 0
            cy = 0
            for p in contour:
                cx += p[0][0]
                cy += p[0][1]
            cx = int(cx/len(contour))
            cy = int(cy/len(contour))
        elif len(contour) != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            continue

        cls_arr = y_cls[iy, ix]
        val, cnt = np.unique(cls_arr, return_counts=True)
        cls = val[np.argmax(cnt)]
        prob = y_prob[iy, ix][cls_arr == cls].mean()

        detections.append((cx, cy, cls, prob, len(contour)))

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', required=True, help="Path to the directory where the model checkpoints are saved")
    parser.add_argument('--split-file', help="Path to json file with train-test split ids. (Optional)")
    parser.add_argument('-v', default=False, type=bool, help="Verbose for dataloading")
    parser.add_argument('--data-path', required=True, help="Path to the training data directory.")
    parser.add_argument('--model-name', help="Name of the model to be used for calculating predictions from the result directory")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_path = args.data_path
    result_path = args.result_dir
    model_name = args.model_name
    split_file = args.split_file
    cv_split = {"full":{"train":[], "test":[]}}
    if split_file:
        cv_split = json.load(open(split_file))

    outer_pbar = tqdm(cv_split.keys(), position=0)

    for cv in outer_pbar:
        os.makedirs(os.path.join(result_path, cv), exist_ok=True)
        cv_path = os.path.join(result_path, cv)

        transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ignore_id =  cv_split[cv]["train"]

        val_data = XView3Data(ignore_id=ignore_id, background_chip_ratio=1.1, obj_size=5, threshold=0.25, overwrite=False,
                            labels_path=None, data_path=data_path, preprocess_label=preprocess_label)

        model_path = os.path.join(cv_path, "ckpts")
        dn = torch.nn.Sequential(*[FeatureExtractor(load_model_config(prepare_backbone())), DN(128, 4)])
        dn.load_state_dict(torch.load(os.path.join(model_path, model_name)))
        dn.to(device)
        dn.eval()


        predictions={
            "center_x":[],
            "center_y":[],
            "scene_id":[],
            "class":[],
            "prob":[],
            "scene_h":[],
            "scene_w":[],
            "area":[]
        }

        dataset = val_data

        l = len(dataset.data_ixs)
        for i, uid in enumerate(dataset.data_ixs, 1):
            scene_id, chip_row, chip_col = uid.split("$")
            chip_row, chip_col = int(chip_row), int(chip_col)

            scene = dataset.scenes[scene_id]
            rgb_img, flag = scene[chip_row, chip_col]

            (strt_r, strt_c), _ = scene.get_full_index(chip_row, chip_col)

            with torch.no_grad():
                rgb_img = torch.from_numpy(np.expand_dims(rgb_img, 0))

                img = transform(rgb_img)
                class_pred = dn(img.to(device))
                detections = get_detections(class_pred[0])

                print(f"\r{i}/{l} processing... got detections: {len(detections)}  ", end="")

                for (center_x, center_y, cls, prob, area) in detections:
                    center_x += strt_c
                    center_y += strt_r

                    predictions["center_x"].append(center_x)
                    predictions["center_y"].append(center_y)
                    predictions["scene_id"].append(scene_id)
                    predictions["class"].append(cls)
                    predictions["prob"].append(prob)
                    predictions["scene_h"].append(scene.row)
                    predictions["scene_w"].append(scene.col)
                    predictions["area"].append(area)

        df = pd.DataFrame(predictions)
        df.to_csv(os.path.join(cv_path, f"results.csv"), index=False)

