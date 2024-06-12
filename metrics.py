from pathlib import Path
import numpy as np
import cv2

img_dir = Path('false_positive/fp_detection/images_w_border')
student_labels_dir = Path('false_positive/fp_detection/students_labels')
yolo_labels_dir = Path('false_positive/fp_detection/yolo_labels/labels')
output_dir_fp = Path('false_positive/fp_detection/false_positives')
output_dir_fn = Path('false_positive/fp_detection/false_negatives')
fp_annotations_dir = Path('false_positive/fp_detection/fp_annotations')
fn_annotations_dir = Path('false_positive/fp_detection/fn_annotations')
output_dir_gt = Path('false_positive/fp_detection/ground_truth')

# IOU and confidence thresholds
IOU_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.24


def load_annotations(file_path, with_confidence=False):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            if with_confidence:
                class_id, x_center, y_center, width, height, confidence = parts
                annotations.append([class_id, x_center, y_center, width, height, confidence])
            else:
                class_id, x_center, y_center, width, height = parts
                annotations.append([class_id, x_center, y_center, width, height])
    return annotations


def convert_yolo_to_xyxy(bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = bbox[:5]
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return [x1, y1, x2, y2, class_id]


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = box1_area + box2_area - intersection
    return intersection / union


def draw_and_save_boxes(img_path, boxes, output_path, color, label):
    img = cv2.imread(str(img_path))
    img_height, img_width = img.shape[:2]
    for box in boxes:
        class_id, x_center, y_center, width, height = box[:5]
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Draw box
        cv2.putText(img, f"{label} {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(str(output_path), img)


def save_annotations(annotations, output_dir):
    for img_path, boxes in annotations:
        annotation_path = output_dir / img_path.with_suffix('.txt').name
        with open(annotation_path, 'w') as f:
            for box in boxes:
                class_id, x_center, y_center, width, height = box[:5]
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


false_positives = []
true_positives_count = 0
false_negatives_count = 0
fp_annotations = []
fn_annotations = []
gt_annotations = []

for img_path in img_dir.glob('*.jpg'):
    img_name = img_path.name
    student_ann_path = student_labels_dir / img_name.replace('.jpg', '.txt')
    yolo_ann_path = yolo_labels_dir / img_name.replace('.jpg', '.txt')

    # Load ground truth annotations
    if student_ann_path.exists():
        gt_boxes = load_annotations(student_ann_path)
    else:
        gt_boxes = []

    # Load YOLO predictions with confidence
    if yolo_ann_path.exists():
        pred_boxes = load_annotations(yolo_ann_path, with_confidence=True)
    else:
        pred_boxes = []

    # Load image to get dimensions
    img = cv2.imread(str(img_path))
    img_height, img_width = img.shape[:2]

    # Convert ground truth boxes to xyxy format
    gt_boxes_xyxy = [convert_yolo_to_xyxy(box, img_width, img_height) for box in gt_boxes]

    # Convert predicted boxes to xyxy format, keeping the confidence score
    pred_boxes_xyxy = [convert_yolo_to_xyxy(box, img_width, img_height) + [box[5]] for box in pred_boxes]

    fp_boxes = []
    fn_boxes = []

    # Compare predictions with ground truth
    matched_gt_boxes = set()
    for pred_box in pred_boxes_xyxy:
        x1, y1, x2, y2, cls, conf = pred_box
        iou_max = 0
        matched_gt_box = None
        for gt_box in gt_boxes_xyxy:
            if int(cls) == int(gt_box[4]):
                iou = calculate_iou([x1, y1, x2, y2], gt_box[:4])
                if iou > iou_max:
                    iou_max = iou
                    matched_gt_box = gt_box

        if iou_max > IOU_THRESHOLD and conf > CONFIDENCE_THRESHOLD:
            true_positives_count += 1
            matched_gt_boxes.add(tuple(matched_gt_box))
        else:
            fp_boxes.append([cls, (x1 + x2) / 2 / img_width, (y1 + y2) / 2 / img_height, (x2 - x1) / img_width,
                             (y2 - y1) / img_height, conf])

    # Identify false negatives
    for gt_box in gt_boxes_xyxy:
        if tuple(gt_box) not in matched_gt_boxes:
            false_negatives_count += 1
            fn_boxes.append(
                [gt_box[4], (gt_box[0] + gt_box[2]) / 2 / img_width, (gt_box[1] + gt_box[3]) / 2 / img_height,
                 (gt_box[2] - gt_box[0]) / img_width, (gt_box[3] - gt_box[1]) / img_height])
            output_path = output_dir_fn / img_name
            draw_and_save_boxes(img_path, fn_boxes, output_path, (255, 0, 0), "FN")

    if fp_boxes:
        false_positives.append((img_path, gt_boxes_xyxy, pred_boxes_xyxy))
        fp_annotations.append((img_path, fp_boxes))
        output_path = output_dir_fp / img_name
        draw_and_save_boxes(img_path, fp_boxes, output_path, (0, 0, 255), "FP")

    if fn_boxes:
        fn_annotations.append((img_path, fn_boxes))

    # Draw and save ground truth annotations
    if gt_boxes:
        output_path = output_dir_gt / img_name
        #draw_and_save_boxes(img_path, gt_boxes, output_path, (0, 255, 0), "GT")
        gt_annotations.append((img_path, gt_boxes))

save_annotations(fp_annotations, fp_annotations_dir)
save_annotations(fn_annotations, fn_annotations_dir)

print(f"Total True Positives: {true_positives_count}")
print(f"Total False Negatives: {false_negatives_count}")
print(f"Total False Positives: {len(fp_annotations)}")

