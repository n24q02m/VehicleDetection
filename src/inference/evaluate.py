import os
import sys
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

def overlap(gt, q):
    q = q.copy() 
    gt = gt.copy()
    q['xmax'] = q['xmin'] + q['width']
    q['ymax'] = q['ymin'] + q['height']
    
    gt['xmax'] = gt['xmin'] + gt['width']
    gt['ymax'] = gt['ymin'] + gt['height']
    
    area = q['width'] * q['height']
    
    iw = (gt['xmax'].clip(upper=q['xmax']) - gt['xmin'].clip(lower=q['xmin'])).clip(lower=0.)
    ih = (gt['ymax'].clip(upper=q['ymax']) - gt['ymin'].clip(lower=q['ymin'])).clip(lower=0.)
    
    ua = (gt['width'] * gt['height']) + area - (iw * ih)
    
    return (iw * ih / ua).fillna(0.)


def ap(tp, m):
    fp = np.zeros(tp.size)
    for i in range(tp.size):
        if tp[i] == 0:
            fp[i] = 1
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / m
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    mpre = np.concatenate(([0.], prec, [0.]))
    mrec = np.concatenate(([0.], rec, [1.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def process_row(args):
    
    gt, r = args

    # print(gt.shape)
    # print(r.shape)

    # Compute IoU only if ground-truth data is not empty
    if gt.empty:
        return None

    cvfgt = gt.assign(iou=overlap(gt, r))
    # print(cvfgt)

    # Check if IoU calculation resulted in an empty or invalid DataFrame
    if cvfgt.empty or cvfgt['iou'].isna().all():
        return None

    best_match = cvfgt.loc[cvfgt.iou.idxmax()]

    if best_match.iou >= 0.5 and not best_match.selected:
        return best_match.name, r.name  # Return indices for updates

    return None


def process_class_data(gt, pr, cid):

    # gt, pr, cid = args
    cgt = gt[gt.cid == cid]
    cpr = pr[pr.cid == cid]
    m = len(cgt)

    if cpr.empty or cgt.empty:
        return 0

    # Prepare tasks for parallel processing of each prediction row
    tasks = [
        (cgt[cgt.fid == fid], r)
        for fid in cpr.fid.unique()
        for _, r in cpr[cpr.fid == fid].iterrows()
    ]


    with Pool(cpu_count()) as pool:
        results = pool.map(process_row, tasks)


    # Apply updates based on results
    for result in results:
        if result:
            gt_idx, pr_idx = result
            if not gt.loc[gt_idx, 'selected']:
                gt.loc[gt_idx, 'selected'] = True
                pr.loc[pr_idx, 'tp'] = 1

    return ap(pr[pr.cid == cid].tp.values, m)


def compute_map(gt, pr):
    gt = gt.assign(selected=False)
    pr = pr.sort_values(by='confidence', ascending=False).assign(tp=0)

    class_ids = gt.cid.unique()
    results = [process_class_data(gt, pr, cid) for cid in class_ids]

    return np.mean(results)

def read_boxes(filename, is_gt=True):
    boxes = set()
    with open(filename, 'r', encoding='utf8') as fIn:
        for line in fIn:
            parts = line.strip().split()
            image_name = parts[0]
            class_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            if is_gt:
                boxes.add((image_name, class_id, x, y, w, h))
            else:
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                boxes.add((image_name, class_id, x, y, w, h, confidence))
    return list(boxes)

if __name__ == "__main__":

    # python scoring/scoring/evaluate.py  test/input/ test/output/

    [_, input_dir, output_dir] = sys.argv
    import logging
    logger = logging.getLogger(__name__)

    submission_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')
    truth_file = os.path.join(truth_dir, 'public_gt/gt.txt')
    submission_answer_file = os.path.join(submission_dir, 'predict.txt')

    if not os.path.exists(submission_answer_file):
        raise Exception("There is no predict.txt file")


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')

    gt_boxes = read_boxes(truth_file, is_gt=True)
    try:
        pred_boxes = read_boxes(submission_answer_file, is_gt=False)
    except Exception as e:
        raise Exception("Every line should be in the format of ('image_name', 'class_id', 'xmin', 'ymin', 'width', 'height', 'confidence')")

    gt_df = pd.DataFrame(gt_boxes, columns=['image_name', 'class_id', 'xmin', 'ymin', 'width', 'height'])
    pred_df = pd.DataFrame(pred_boxes, columns=['image_name', 'class_id', 'xmin', 'ymin', 'width', 'height', 'confidence'])

    gt_df.rename(columns={'image_name': 'fid', 'class_id': 'cid'}, inplace=True)
    pred_df.rename(columns={'image_name': 'fid', 'class_id': 'cid'}, inplace=True)

    mAP_score = compute_map(gt_df, pred_df)
    

    with open(output_filename, 'w') as output_file:
        output_file.write(f"MAP: {round(mAP_score, 4)}\n")
    
    sys.stdout.write(f"MAP: {round(mAP_score, 4)}\n")
