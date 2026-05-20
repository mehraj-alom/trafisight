from config.global_config import GlobalConfig
from logger import inference_logger as logger
from utils.global_utils import load_onnx_session
from detection.detector_utils import preprocess_frame, postprocess_detections


class Detector:
	def __init__(self, model_path: str = GlobalConfig.VEHICLE_MODEL_ONNX):
		try:
			self.session = load_onnx_session(model_path)
			self.input_name = self.session.get_inputs()[0].name
			self.input_shape = self.session.get_inputs()[0].shape
			logger.info(f"Loaded detector model from: {model_path}")
		except Exception as e:
			logger.error(f"Failed to load detector model '{model_path}': {e}")
			raise

	def detect_vehicle(self, frame) :
		"""Run detection on a single BGR frame and return postprocessed detections.

		Returns an empty list on error so callers can continue processing frames.
		"""
		try:
			input_tensor = preprocess_frame(frame, self.input_shape)
			outputs = self.session.run(None, {self.input_name: input_tensor})
			detections = postprocess_detections(
				outputs[0], frame.shape, (input_tensor.shape[2], input_tensor.shape[3]),
			)
			return detections
		except Exception as e:
			logger.error(f"Error during vehicle detection: {e}")
			return []

