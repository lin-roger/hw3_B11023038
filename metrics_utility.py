import numpy as np
import cv2 as cv
from segmentation_models_pytorch.metrics import iou_score, get_stats

def keep_largest_seg(img):
    copy_img = img.copy()
    contours, _ = cv.findContours(copy_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        masks = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[1:]
        cv.drawContours(copy_img, masks, -1, 0, thickness=cv.FILLED)
    return copy_img

def post_process(preds, labels):
    preds = (preds.cpu().numpy() > 0.5).astype(np.uint8) * 255
    preds = np.stack([np.expand_dims(keep_largest_seg(i[0]), axis=0) for i in preds], axis=0)
    labels = labels.cpu().numpy().astype(np.uint8) * 255
    return preds, labels


def find_bottom(preds):
    tmp = []
    for i in range(len(preds)):
        img = preds[i][0]
        height, _ = img.shape
        tmp.append(height-1)
        for y in range(height - 1, -1, -1):
            if np.any(img[y] == 255):
                tmp[-1] = y
                break
    return np.array(tmp)


class MetricsCalculator:
    def __init__(self, pixels_per_cm, scale):
        self.pixels_per_cm = pixels_per_cm/scale
        self.num_samples = 0
        self.total_iou = 0
        self.total_error_cm = 0
        self.num_of_error_less_than_50mm = 0
        self.num_of_error_less_than_1cm = 0

    def _compute_iou(self, preds, labels):
        tp, fp, fn, tn = get_stats(
            preds, labels, mode="binary", threshold=0.5, num_classes=1
        )
        iou = iou_score(tp, fp, fn, tn, reduction="micro")
        return iou

    def __call__(self, preds, labels):
        nos = len(preds)
        self.num_samples += nos
        self.total_iou += self._compute_iou(preds, labels).cpu().numpy() * nos
        preds, labels = post_process(preds, labels)
        preds_bottom, labels_bottom = find_bottom(preds), find_bottom(labels)
        d = np.abs(preds_bottom - labels_bottom)
        self.total_error_cm += np.mean(d) / self.pixels_per_cm * nos
        self.num_of_error_less_than_50mm += np.sum(d < 0.5 * self.pixels_per_cm)
        self.num_of_error_less_than_1cm += np.sum(d < 1 * self.pixels_per_cm)

    def compute(self):
        if self.num_samples == 0:
            raise ValueError("No samples to compute metrics.")
        avg_iou = self.total_iou / self.num_samples
        avg_error_cm = self.total_error_cm / self.num_samples
        avg_error_0_5cm = self.num_of_error_less_than_50mm / self.num_samples * 100
        avg_error_1cm = self.num_of_error_less_than_1cm / self.num_samples * 100
        return {
            "iou": avg_iou,
            "error_cm": avg_error_cm,
            "error_0_5cm": avg_error_0_5cm,
            "error_1cm": avg_error_1cm,
        }
