import numpy as np
from modules.config import label_dict
import torch
import cv2

def contour_extract(proposal_t, thresholds):
    proposal = proposal_t.detach().cpu().numpy().squeeze(1)
    indexes = []
    for thresh in thresholds:
        b, i, j = np.where(proposal > thresh)
        indexes.append((b, i, j))
    
    return indexes

def plt_img(tensor):
    
    img_arr = tensor.detach().cpu().numpy()
    return np.transpose(img_arr, (1,2,0))
    

def convert_prob_to_image(prob, pred=False):
    prob = prob.detach().cpu().numpy()
    h, w  = prob.shape[1:] if pred else prob.shape[:2]
    class_labels_rgb = np.empty((h, w, 3), dtype=np.uint8)
    
    if pred:
        prob = prob.transpose(1,2,0)
        prob = np.argmax(prob, axis=2)
    
    for val in np.unique(prob):
        rows, cols = np.where(prob == val)
        if val == 0:
            color = (0, 0, 0)
        elif val == 1: #vessel-RED or vessel-fishing-RED
            color = (255, 0, 0)
        elif val == 2: #not_vessel-GREEN or vessel-not-fishing-GREEN
            color = (0, 255, 0)
        elif val == 3: #not_vessel-BLUE
            color = (0, 0, 255)
        class_labels_rgb[rows, cols] = color
    
    
    class_labels_rgb = class_labels_rgb.transpose(2, 0, 1)
    return torch.from_numpy(class_labels_rgb)


def save_fig(writer, split, step, img, class_pred, class_true, multiclass=False):
    

    writer.add_image(f'{split}_input_image', img, global_step=step)
    
    if multiclass:
        writer.add_image(f'{split}_class_prediction', convert_prob_to_image(class_pred, pred=True), global_step=step)
        
        writer.add_image(f'{split}_class_true', convert_prob_to_image(class_true), global_step=step)
    else:
        writer.add_image(f'{split}_class_prediction', class_pred, global_step=step)
        
        writer.add_image(f'{split}_class_true', class_true, global_step=step)


def evaluate(model, dataloader, criterion, step, total_size, device, writer, transform=None, multiclass=False):
    
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 1):
            if i > total_size:
                break
            if multiclass:
                proposal_t, un_img, class_labels, inst_weight = [torch.cat(d, dim=0) for d in data]
            else:
                un_img, class_labels, inst_weight = [torch.cat(d, dim=0) for d in data]
            
            if transform is not None:
                img = transform(un_img)
            img = img.to(device)
            
            if not multiclass:
                class_labels = class_labels.cpu().numpy()
                class_labels[class_labels != 0] = 1
                class_labels = torch.from_numpy(class_labels.astype("float32")).unsqueeze(1).to(device)
            else:
                class_labels = class_labels.to(device)

            pred = model(img)
            if multiclass:
                loss = criterion(proposal_t, pred, class_labels)
            else:
                loss = criterion(pred, class_labels)
            
            if torch.isnan(loss):
                continue
            val_loss += loss.detach().item()
        if step is not None:
                save_fig(writer, "val", step, un_img[0], pred[0], class_labels[0], multiclass=multiclass)
    return val_loss/total_size




def adaptive_gaussian_kernel(x, y, mu_x, mu_y, sigma, chipsize=800):
    mu_x -= chipsize//2
    mu_y -=chipsize//2
    return (np.exp(-1*((x-mu_x)**2 + (y-mu_y)**2)/(2*sigma*sigma))*(1/np.sqrt(2*np.pi)*sigma)).reshape(chipsize,chipsize).astype(np.float32)



def prepare_label(labels, chipsize, obj_size, threshold, fishing=False, length=False):
    x, y = np.meshgrid(np.arange(-chipsize/2, chipsize/2, 1), np.arange(-chipsize/2, chipsize/2, 1))
    x, y = x.reshape(-1), y.reshape(-1)

    inst_weight = np.zeros((chipsize, chipsize), dtype=np.float32)
    class_labels = np.zeros((chipsize, chipsize), dtype=np.int32)
    length_arr = -1*np.ones((chipsize, chipsize), dtype=np.float32)
    
    if labels is None:
        if length:
            return class_labels, inst_weight, length_arr
        else:
            return class_labels, inst_weight

    for row, col, is_vessel, is_fishing, vessel_length in zip(labels["detect_scene_row"], labels["detect_scene_column"], labels["is_vessel"], labels["is_fishing"], labels["vessel_length_m"]):
        k = adaptive_gaussian_kernel(x, y, mu_x=col, mu_y=row, sigma=obj_size)
        k /= k.max()
        class_mask = np.where(k>threshold)
        
        if fishing:
            if is_vessel:
                if is_fishing:
                    label = label_dict["vessel_fishing"]
                else:
                    label = label_dict["vessel_not_fishing"]
            else:
                label = label_dict["vessel_not"]
        else:
            label = label_dict["vessel"] if is_vessel else label_dict["not_vessel"]

        class_labels[class_mask[0], class_mask[1]] = label
        if is_vessel and not np.isnan(vessel_length):
            length_arr[class_mask[0], class_mask[1]] = vessel_length

        inst_weight += k
    
    row_ix, col_ix = np.where(inst_weight <= threshold)
    inst_weight[row_ix, col_ix] = 1.0
    if length:
        return class_labels, inst_weight, length_arr
    else:
        return class_labels, inst_weight


def joint_prob(fa, sr):
    
    joint_prob = torch.clone(sr)
    joint_prob[:, 0, :, :] = (1-fa).squeeze(dim=1) * sr[:, 0, :, :]
    joint_prob[:, 1:, :, :] = fa * sr[:, 1:, :, :]
    
    Z = torch.sum(joint_prob, dim=1, keepdim=True)
    
    p_ci = joint_prob / Z

    return p_ci


    
def convertRGB(vv, vh):

    r, c = vv.shape
    img = np.empty((3, r, c), dtype=np.float32)
    
    if np.max(vv) == np.min(vv):
        return np.zeros_like(img), True
    
    if np.max(vh) == np.min(vh):
        return np.zeros_like(img), True
    
    b = vv - vh
    
    if np.max(b) == np.min(b):
        return np.zeros_like(img), True

    img[0, :, :] = vv
    img[1, :, :] = vh
    img[2, :, :] = b
    
    img_min = img.min(axis=(1,2)).reshape(3, 1, 1)
    img_max = img.max(axis=(1,2)).reshape(3, 1, 1)
    
    
    return (img - img_min)/(img_max - img_min), False



def collate_fn(batch):
    return tuple(zip(*batch))




def get_detections(joint_prob, num_classes = 2):

    detections = []
    y_arr = joint_prob.detach().cpu().numpy()

    y_cls = np.argmax(y_arr, axis=0)
    y_prob = y_arr.max(axis=0)

    for i in range(1, num_classes+1):
        cls = i
        y_bin = (y_cls == i).astype(np.uint8)

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
            detections.append((cx, cy, cls, y_prob[iy, ix].mean()))
        
    return detections
            

def get_detections_fa(fa, threshold=0.37):

    detections = []
    y_arr = fa.detach().cpu().numpy()

    y_bin = (y_arr[0] >= threshold).astype(np.uint8)

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
        detections.append((cx, cy, 1, y_arr[0][iy, ix].mean()))
        
    return detections

