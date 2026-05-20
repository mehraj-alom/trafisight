from pathlib import Path
import importlib

try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

class GlobalConfig:
    # Paths 
    WEIGHTS_DIR = Path("detection/training/vehicle_detection/weights")
    OUTPUT_PATH = Path("outputs") / "inference"
    
    
    # MODEL Settings
    VEHICLE_MODEL_ONNX = str(WEIGHTS_DIR / "best.onnx")
    VEHICLE_MODEL_PT = str(WEIGHTS_DIR / "best.pt")

    # Backward-compatible aliases for older references.
    Vehicle_MODEL_ONNX = VEHICLE_MODEL_ONNX
    Vehicle_MODEL_PT = VEHICLE_MODEL_PT
    
    # Clasd Names 
    CLASS_NAMES = ["animal", "autorickshaw", "bicycle", "bus", "car", "caravan", "motorcycle", "person", "rider", "traffic light", "traffic sign", "trailer", "train", "truck", "vehicle fallback"]
    CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    