# import onnxruntime as ort
# from config.global_config import GlobalConfig
# import numpy as np
# from utils.global_utils import load_onnx_session

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


# img = cv2.imread("images.jpeg")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_rgb = cv2.resize(img_rgb, (640, 640))
# tensor = img_rgb.astype(np.float32) / 255.0
# tensor = np.transpose(tensor, (2, 0, 1))
# tensor = np.expand_dims(tensor, axis=0)
# outputs = session.run(
#     None,
#     {
#         session.get_inputs()[0].name: tensor
#     }
# )

# preds = np.squeeze(outputs[0]).T

# class_scores = preds[:, 4:]

# best_idx = np.argmax(
#     class_scores.max(axis=1)
# )

# best_pred = preds[best_idx]

# print("BEST PREDICTION:")
# print(best_pred)

# print("CLASS ID:")
# print(np.argmax(best_pred[4:]))

# print("SCORES:")
# print(best_pred[4:])

from pathlib import Path

import cv2
import numpy as np

from logger import inference_logger as logger
from utils.global_utils import load_onnx_session
from config.global_config import GlobalConfig


class HelmetDetector:

    HELMET = 0
    NO_HELMET = 1
    BIKE = 2

    def __init__(
        self,
        model_path: str = GlobalConfig.HELMET_MODEL_ONNX,
        conf_threshold: float = 0.50,
    ) -> None:

        self.model_path = model_path
        self.conf_threshold = conf_threshold

        self.enabled = False
        self.session = None
        self.input_name = None
        self.input_shape = None

        try:

            if not Path(model_path).exists():
                logger.warning(
                    f"Helmet model not found: {model_path}"
                )
                return
            self.session = load_onnx_session(
                model_path
            )
            self.input_name = (
                self.session.get_inputs()[0].name
            )
            self.input_shape = (
                self.session.get_inputs()[0].shape
            )
            self.enabled = True
            logger.info(
                f"Loaded helmet model: {model_path}"
            )

        except Exception as exc:
            logger.warning(
                f"Helmet detector disabled: {exc}"
            )


    def detect(self, frame):

        if (
            not self.enabled
            or self.session is None
            or self.input_shape is None
        ):
            return None

        try:

            input_tensor = (
                self._preprocess_frame(frame)
            )

            outputs = self.session.run(
                None,
                {
                    self.input_name:
                    input_tensor
                }
            )

            return self._postprocess(
                outputs[0]
            )

        except Exception as exc:

            logger.warning(
                f"Helmet detection failed: {exc}"
            )

            return None

    def _preprocess_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        height = (
            self.input_shape[2]
            if isinstance(
                self.input_shape[2],
                int
            )
            else 640
        )

        width = (
            self.input_shape[3]
            if isinstance(
                self.input_shape[3],
                int
            )
            else 640
        )

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (width, height)
        )

        image = (
            image.astype(np.float32)
            / 255.0
        )

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        return image

    def _postprocess(
        self,
        raw_output: np.ndarray,
    ):

        output = np.squeeze(
            raw_output
        )
        if output.ndim != 2:
            return None
        predictions = output.T
        best_class = None
        best_score = 0.0
        for pred in predictions:

            scores = pred[4:]

            class_id = int(
                np.argmax(scores)
            )

            score = float(
                scores[class_id]
            )

            if score < self.conf_threshold:
                continue
            if score > best_score:
                best_score = score
                best_class = class_id

        if best_class is None:
            return None
        
        logger.info(
            f"Helmet class: "
            f"{best_class}, "
            f"score={best_score:.3f}"
        )

        return best_class