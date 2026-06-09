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
    LOCAL_DEBUG_PATH = OUTPUT_PATH / "local_debug"
    
    # MODEL Settings
    VEHICLE_MODEL_ONNX = str(WEIGHTS_DIR / "best.onnx")
    VEHICLE_MODEL_PT = str(WEIGHTS_DIR / "best.pt")

    # OCR / Track-memory Settings
    OCR_TRACK_EXPIRY_FRAMES = 120
    OCR_MAX_ATTEMPTS_PER_TRACK = 3

    # Backward-compatible aliases for older references.
    Vehicle_MODEL_ONNX = VEHICLE_MODEL_ONNX
    Vehicle_MODEL_PT = VEHICLE_MODEL_PT

    #Helmet Detection Model Path
    HELMET_MODEL_ONNX = "detection/helmet_detection/helmet.onnx"
    
    # Clasd Names 
    CLASS_NAMES = ["animal", "autorickshaw", "bicycle", "bus", "car", "caravan", "motorcycle", "person", "rider", "traffic light", "traffic sign", "trailer", "train", "truck", "vehicle fallback"]
    CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    