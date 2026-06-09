import onnxruntime as ort
# import numpy as np

MODEL_PATH = "detection/helmet_detection/helmet.onnx"  

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

# print("=== INPUTS ===")
# for inp in session.get_inputs():
#     print(inp.name, inp.shape)

# print("\n=== OUTPUTS ===")
# for out in session.get_outputs():
#     print(out.name, out.shape)

# input_name = session.get_inputs()[0].name
# dummy = np.random.rand(
#     1,
#     3,
#     640,
#     640
# ).astype(np.float32)
# outputs = session.run(
#     None,
#     {
#         input_name: dummy
#     }
# )
# print("\nOutput shape:")
# print(outputs[0].shape)
# #output form
# output = outputs[0]
# preds = np.squeeze(output)

# print("\nAfter squeeze:")
# print(preds.shape)

# print("\nFirst prediction:")
# if preds.ndim == 2:
#     print(preds.T[0][:20] if preds.shape[0] < preds.shape[1] else preds[0][:20])
# else:
#     print(preds[:20])




# === INPUTS ===
# images ['batch', 3, 'height', 'width']

# === OUTPUTS ===
# output0 ['batch', 7, 'anchors']

# Output shape:
# (1, 7, 8400)

# After squeeze:
# (7, 8400)

# First prediction:
# [5.5638466e+00 1.2606810e+01 1.0662094e+01 2.5290817e+01 1.4065206e-03
#  1.6906857e-04 8.8572502e-05]




# output = outputs[0]
# preds = np.squeeze(output).T
# print(preds.shape)
# for i in range(5):
#     print(preds[i])

                                                  ########## Output shape:
# (1, 7, 8400)
# (8400, 7)      
# [6.3585987e+00 1.4771431e+01 1.2252256e+01 2.9628986e+01 6.6190660e-03
#  3.4186244e-04 1.7198920e-04]
# [1.2140515e+01 1.1358717e+01 2.3079840e+01 2.3094269e+01 2.2043288e-03
#  1.7711520e-04 1.0544062e-04]
# [2.4494171e+01 5.5236673e+00 3.1663254e+01 1.1216349e+01 1.0714829e-03
#  1.3378263e-04 3.9815903e-05]
# [2.8125544e+01 5.4740849e+00 2.6122456e+01 1.1031601e+01 1.0877550e-03
#  1.5443563e-04 2.6673079e-05]
# [3.5562325e+01 6.2365117e+00 2.1847164e+01 1.2715550e+01 1.2054443e-03
#  2.0095706e-04 2.4288893e-05]
# (venv) mehrajalom0@penguin:~/trafisight$ 


import cv2
import numpy as np

img = cv2.imread("images.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = cv2.resize(img_rgb, (640, 640))
tensor = img_rgb.astype(np.float32) / 255.0
tensor = np.transpose(tensor, (2, 0, 1))
tensor = np.expand_dims(tensor, axis=0)
outputs = session.run(
    None,
    {
        session.get_inputs()[0].name: tensor
    }
)

preds = np.squeeze(outputs[0]).T

class_scores = preds[:, 4:]

best_idx = np.argmax(
    class_scores.max(axis=1)
)

best_pred = preds[best_idx]

print("BEST PREDICTION:")
print(best_pred)

print("CLASS ID:")
print(np.argmax(best_pred[4:]))

print("SCORES:")
print(best_pred[4:])