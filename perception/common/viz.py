"""Draw detection boxes for the optional --show preview window."""
import cv2


def draw_detections(img, detections):
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        label = f'{d.get("name", d.get("cls"))} {d.get("conf", 0):.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img
