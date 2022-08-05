import glob
import os
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torchvision


coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


def inference_fasterrcnn(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


def val(target_class = ["car", "bus", "truck"], image_dir="./gen_results"):
    device = "cuda:0"

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)

    p = str(Path(image_dir).resolve())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    
    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    sum = 0
    fails = 0

    for i, image_path in enumerate(tqdm(images)):
        sum += 1

        image = np.array(Image.open(image_path))
        image_float_np = np.float32(image) / 255
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        input_tensor = transform(image_float_np)
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor.unsqueeze(0)

        boxes, classes, labels, indices = inference_fasterrcnn(input_tensor, model, device, 0.6)


        if i % 10 == 0:
            image = draw_boxes(boxes, labels, classes, image)
            # Show the image:
            Image.fromarray(image).save(f"val_saves/test_{i}.png")


        for c in classes:
            if c in target_class:
                fails += 1
                break
        
        # image = draw_boxes(boxes, labels, classes, image)

        # # Show the image:
        # Image.fromarray(image).save("test.png")

            
    
    print(f"evaulate result: {sum - fails} / {sum}")
    print(f"attack success rate {1 - fails / sum}")


if __name__ == '__main__':
    val()