from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def _letterbox(image, new_shape=(1280, 1280), color=(114, 114, 114)):
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, r, (dw, dh)


def _xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _postprocess_raw(pred, conf, iou, ratio, dwdh, orig_shape):
    # YOLOv8 ONNX raw output: (1, 4 + nc, n) -> (n, 4 + nc)
    if pred.ndim == 3:
        pred = np.squeeze(pred, axis=0)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes = pred[:, :4]
    cls_scores = pred[:, 4:]
    if cls_scores.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    scores = cls_scores.max(axis=1)
    cls_ids = cls_scores.argmax(axis=1)
    mask = scores >= conf
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    boxes = _xywh2xyxy(boxes)
    boxes[:, [0, 2]] -= dwdh[0]
    boxes[:, [1, 3]] -= dwdh[1]
    boxes /= ratio

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

    nms_input = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    nms_indices = cv2.dnn.NMSBoxes(nms_input, scores.tolist(), conf, iou)
    if len(nms_indices) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    keep = np.array(nms_indices).reshape(-1)
    return boxes[keep], scores[keep]


def _postprocess_nms(pred, conf, orig_shape):
    # NMS-export format often: (1, N, 6) => [x1, y1, x2, y2, score, cls]
    if pred.ndim == 3:
        pred = np.squeeze(pred, axis=0)
    if pred.shape[-1] < 5:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = pred[:, :4]
    scores = pred[:, 4]
    mask = scores >= conf
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = boxes[mask]
    scores = scores[mask]

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    return boxes, scores


def infer_count(weights_path: str, image_path: str, conf=0.2, iou=0.25, save_vis=True):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: '{image_path}'")

    session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    imgsz = 1280
    if len(input_shape) >= 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
        imgsz = max(input_shape[2], input_shape[3])

    resized, ratio, dwdh = _letterbox(image, new_shape=(imgsz, imgsz))
    x = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    output = session.run(None, {input_name: x})[0]
    if output.ndim == 3 and output.shape[-1] in (6, 7):
        boxes, scores = _postprocess_nms(output, conf, image.shape)
    else:
        boxes, scores = _postprocess_raw(output, conf, iou, ratio, dwdh, image.shape)

    count = int(len(boxes))
    print("方格数量:", count)

    if save_vis:
        vis = image.copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{float(score):.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        out = Path("out")
        out.mkdir(exist_ok=True)
        out_path = out / (Path(image_path).stem + "_vis.png")
        cv2.imwrite(str(out_path), vis)
        print("可视化保存:", out_path)

    return count


if __name__ == "__main__":
    weights = "best.onnx"
    img = "D:/xinjianwenjianjia/11.png"
    infer_count(weights, img)
