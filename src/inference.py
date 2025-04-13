import cv2
import typing
import numpy as np
import random

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.utils.text_utils import get_cer

    # Manually set model path and vocab
    model_path = "Models/1_image_to_word/202504111940/model.h5"
    char_list = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    model = ImageToWordModel(model_path=model_path, char_list=char_list)

    # Load annotation test file
    with open("OCR/data/90kDICT32px/annotation_test.txt", "r") as f:
        lines = f.readlines()

    # Pick 20 random samples
    samples = random.sample(lines, 20)

    accum_cer = []

    for line in samples:
        image_path, label = line.strip().split(" ")
        image_path = image_path.replace("\\", "/")

        try:
            image = cv2.imread(image_path)
            prediction_text = model.predict(image)
            cer = get_cer(prediction_text, label)
            print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
        except Exception as e:
            print(f"Error with {image_path}: {e}")
            continue

        accum_cer.append(cer)

    print(f"\n✅ Average CER on 20 samples: {np.average(accum_cer):.4f}")
