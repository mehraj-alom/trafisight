# test_onnx_info.py

from utils.global_utils import load_onnx_session

MODEL_PATH = "ocr/models/LP_DETECTOR.onnx"

session = load_onnx_session(MODEL_PATH)

print("\n===== INPUTS =====")
for inp in session.get_inputs():
    print("Name :", inp.name)
    print("Shape:", inp.shape)
    print("Type :", inp.type)
    print()

print("\n===== OUTPUTS =====")
for out in session.get_outputs():
    print("Name :", out.name)
    print("Shape:", out.shape)
    print("Type :", out.type)
    print()