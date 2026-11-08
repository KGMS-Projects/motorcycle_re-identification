"""Pre-download every model so the container works offline."""
from ultralytics import YOLO

YOLO("yolov8n-seg.pt")
try:
    from transformers import AutoImageProcessor, AutoModel
    AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    AutoModel.from_pretrained("facebook/dinov2-small")
except Exception as e:
    print("DINOv2 not cached:", e)
try:
    import easyocr
    easyocr.Reader(["en"], gpu=False, verbose=False)
except Exception as e:
    print("EasyOCR not cached:", e)
print("warm-up complete")
