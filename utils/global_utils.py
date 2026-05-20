
from pathlib import Path
import importlib
from logger import inference_logger as logger

def load_onnx_session(model_path: str):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"ONNX model not found at: {model_file}")

    ort = importlib.import_module("onnxruntime")
    session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    logger.info(f"Loaded ONNX model: {model_file}")
    return session