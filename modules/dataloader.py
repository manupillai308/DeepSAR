import os, pandas as pd, math
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch.utils.data import Dataset
import torch
from modules.utils import convertRGB, prepare_label
import json



class Band(object):
    def __init__(self, path):
        self.path = path
        self.src = rasterio.open(self.path)
    
    def read(self, window):
        img = self.src.read(1, window=window)

        img[img < -50] = -50
        return img
    
    def __eq__(self, other):
        return (self.src.height == other.src.height) and (self.src.width == other.src.width)
    
    def read_resize(self, outsize):
        img = self.src.read(
            out_shape=outsize,
            resampling=Resampling.nearest
        )

        img[img < -50] = -50

        return img

class RGBScene(object):

    def __init__(self, path, id, chipsize, overwrite, shore=False, verbose=False):
        if verbose:
            print(f"\tProcessing scene: {id}")

        self.path = path
        self.shore = shore
        self.json_path = "labels.json"
        self.id = id
        self.vv = Band(os.path.join(self.path, "VV_dB.tif"))
        self.vh = Band(os.path.join(self.path, "VH_dB.tif"))
        self.row = self.vv.src.height
        self.col = self.vv.src.width
        self.labels = None
        self.verbose = verbose
        if os.path.exists(os.path.join(self.path, self.json_path)) and not overwrite:
            if verbose:
                print(f"\tLoading labels from json")
            self.labels = json.load(open(os.path.join(self.path, self.json_path)))

        assert self.vv == self.vh, f"Size mismatch for VV & VH bands in scene {self.id}"
        self.chipsize = chipsize

    def get_chip_indexes(self, labels, background_chip_ratio=0.5):

        to_rows = math.ceil(self.row/self.chipsize)
        to_cols = math.ceil(self.col/self.chipsize)

        chip_indexes = []    
        shore_indexes = []
        shore = False
        overwrite = False
        
        if self.labels is None and labels is not None:
            self.labels = {}
            overwrite = True
            
        bg_chips = 0
        fg_chips = 0
        
        for i in range(to_rows):
            for j in range(to_cols):
                uid = "$".join([self.id, str(i), str(j)])
                if labels is not None:
                    if overwrite:
                        chip_labels = self.gather_labels(i, j, labels)
                    else:
                        chip_labels = self.labels.get(uid, None)
                        if chip_labels is None:
                            continue
                    if self.shore and self.check_shore(chip_labels):
                        shore = True
                    if len(chip_labels["detect_id"]) < 1:
                        if fg_chips == 0  or bg_chips/fg_chips >= background_chip_ratio:
                            continue
                        else:
                            bg_chips += 1
                    else:
                        fg_chips += 1

                    if overwrite:
                        self.labels[uid] = chip_labels
                if shore:
                    shore_indexes.append(uid)
                    shore = False
                else:
                    chip_indexes.append(uid)

        if overwrite and labels is not None:
            if self.verbose:
                print(f"\tSaving in {self.json_path}")
            json.dump(self.labels, open(os.path.join(self.path, self.json_path), "w"))
        if self.verbose:
            print(f"\tTotal chips extracted: {len(chip_indexes)}")
        if labels is not None and self.verbose:
            print(f"\t\tBackground chips: {bg_chips}")
            print(f"\t\tForeground chips: {fg_chips}")
        
        if self.shore:
            return chip_indexes, shore_indexes
        else:
            return chip_indexes
    
    def check_shore(self, chip_labels):
        return np.any(np.array(chip_labels["distance_from_shore_km"]) <= 2)
        
    
    def get_full_index(self, chip_row, chip_col):
        
        strt_off_r = self.chipsize * chip_row
        strt_off_c = self.chipsize * chip_col

        return (strt_off_r, strt_off_c), (strt_off_r + self.chipsize, strt_off_c+self.chipsize)


    def __getitem__(self, index):
        
        chip_row, chip_col = index
        
        
        (strt_r, strt_c), (_, _) = self.get_full_index(chip_row, chip_col)

        vv = self.vv.read(window=Window(strt_c, strt_r, self.chipsize, self.chipsize)) # (col_off, row_off, width, height)
        vh = self.vh.read(window=Window(strt_c, strt_r, self.chipsize, self.chipsize))

        pad_rows = self.chipsize - vv.shape[0]
        pad_cols = self.chipsize - vv.shape[1]
        
        assert (pad_rows >= 0 and pad_cols >= 0), f"Padding error for scene:{self.id}, for chip:{(chip_row, chip_col)}"

        vv = np.pad(vv, pad_width=((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)
        vh = np.pad(vh, pad_width=((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)

        return convertRGB(vv, vh)
    
    def gather_labels(self, chip_row, chip_col, labels):
        if labels is None:
            return None
        
        labels = labels[labels["scene_id"] == self.id]
        (strt_r, strt_c), (end_r, end_c) = self.get_full_index(chip_row, chip_col)

        fil1 = (labels["detect_scene_row"] >= strt_r)
        fil2 = (labels["detect_scene_row"] < end_r)
        fil3 = (labels["detect_scene_column"] >= strt_c)
        fil4 = (labels["detect_scene_column"] < end_c)

        labels = labels.loc[fil1 & fil2 & fil3 & fil4]
        labels["detect_scene_row"] = labels["detect_scene_row"].apply(lambda x: x - (math.floor(x/self.chipsize)*self.chipsize))
        labels["detect_scene_column"] = labels["detect_scene_column"].apply(lambda x: x - (math.floor(x/self.chipsize)*self.chipsize))

        return labels.to_dict(orient="list")

    def get_patch_indexes(self, patchsize):
        
        r = math.ceil(self.row/patchsize)
        c = math.ceil(self.col/patchsize)
        ixs = []

        for i in range(r):
            for j in range(c):
                strt_r, strt_c = i*patchsize, j*patchsize
                end_r, end_c = strt_r+patchsize, strt_c+patchsize
                flag = False
                for chip_row in range(strt_r//self.chipsize, end_r//self.chipsize):
                    for chip_col in range(strt_c//self.chipsize, end_c//self.chipsize):
                        if self.labels is not None:
                            labels = self.labels.get(f"{self.id}_{chip_row}_{chip_col}")
                            if labels is not None:
                                flag = True
                if flag:
                        ixs.append(f"{self.id}_{i}_{j}_{strt_r//self.chipsize}_{strt_c//self.chipsize}_{end_r//self.chipsize}_{end_c//self.chipsize}")
        
        return ixs




class XView3Data(Dataset):

    def __init__(self, labels_path, data_path, ignore_id = [], obj_size=5, threshold=0.5, preprocess_label=lambda x:x, chipsize=800, background_chip_ratio=0.5, overwrite=False, shore=False, verbose=False):

        self.data_path = data_path
        self.labels_path = labels_path
        self.scenes = {}
        self.data_ixs = []
        self.chipsize = chipsize
        self.shore = shore
        self.obj_size = obj_size
        self.threshold = threshold
        labels = None
        if labels_path is not None:
            if isinstance(labels_path, list):
                labels = preprocess_label(pd.read_csv(labels_path[0]))
                for lp in labels_path[1:]:
                    labels = pd.concat([labels, preprocess_label(pd.read_csv(lp))])
            else:
                labels = preprocess_label(pd.read_csv(labels_path))
        
        scene_ids = {}
        
        if isinstance(self.data_path, list):
            for dp in self.data_path:
                scene_ids[dp] = list(filter(lambda x: x not in ignore_id, os.listdir(dp)))
        else:
            scene_ids[self.data_path] = list(filter(lambda x: x not in ignore_id, os.listdir(self.data_path)))
        
        if shore:
            self.shore_ixs = []
        if verbose:
            print(f"Total scenes detected: {sum(map(len, scene_ids.values()))}")
        for dp, s_ids in scene_ids.items():
            for scene_id in s_ids: 
                scene = RGBScene(path=os.path.join(dp, scene_id), id=scene_id, chipsize=chipsize, 
                                 overwrite=overwrite, shore=shore)
                indexes = scene.get_chip_indexes(labels, background_chip_ratio)
                if shore:
                    self.shore_ixs.extend(indexes[1])
                    indexes = indexes[0]
                self.data_ixs.extend(indexes)
                self.scenes[scene_id] = scene
        if verbose:
            print(f"Total chips extracted: {len(self.data_ixs)}")
        
    
    def __len__(self):
        return len(self.data_ixs)
    
    def _getitem__proxy(self, uid):
        
        scene_id, chip_row, chip_col = uid.split("$")
        chip_row, chip_col = int(chip_row), int(chip_col)
        rgb_img, flag = self.scenes[scene_id][chip_row, chip_col]
        if self.scenes[scene_id].labels is None:
            return torch.from_numpy(rgb_img), None, None
        labels = self.scenes[scene_id].labels[uid]

        class_labels, inst_weight = prepare_label(labels, self.chipsize, self.obj_size, self.threshold)
        return torch.from_numpy(rgb_img), torch.from_numpy(class_labels) if not flag else torch.from_numpy(np.zeros_like(class_labels)), torch.from_numpy(inst_weight) if not flag else torch.from_numpy(np.ones_like(inst_weight))
        

    def __getitem__(self, index):
        
        if self.shore:
            return tuple(zip(self._getitem__proxy(self.data_ixs[index]), 
                             self._getitem__proxy(self.shore_ixs[index%len(self.shore_ixs)])))
        else:
            return self._getitem__proxy(self.data_ixs[index])
        
class XView3DDN(XView3Data):

    def __init__(self, model_callable, *args, **kwargs):

        super(XView3DDN, self).__init__(*args, **kwargs)
        self.model_callable = model_callable
        
    def __len__(self):
        return super(XView3DDN, self).__len__()

    def _getitem__proxy(self, uid):
        scene_id, chip_row, chip_col = uid.split("$")
        chip_row, chip_col = int(chip_row), int(chip_col)
        rgb_img, flag = self.scenes[scene_id][chip_row, chip_col]
        rpn = self.model_callable(torch.from_numpy(rgb_img).unsqueeze(dim=0))
        if self.scenes[scene_id].labels is None:
            return rpn[0], torch.from_numpy(rgb_img), None, None
        labels = self.scenes[scene_id].labels[uid]

        class_labels, inst_weight = prepare_label(labels, self.chipsize, self.obj_size, self.threshold, fishing=True)
        return rpn[0], torch.from_numpy(rgb_img), torch.from_numpy(class_labels) if not flag else torch.from_numpy(np.zeros_like(class_labels)), torch.from_numpy(inst_weight) if not flag else torch.from_numpy(np.ones_like(inst_weight))