from typing import Dict, List, Tuple

print('base sssion 1')
import numpy as np
print('base sssion 2')

import onnxruntime as ort
print('base sssion 3')

from PIL import Image
print('base sssion 4')

from PIL.Image import Image as PILImage
print('base sssion 5')



class BaseSession:
    def __init__(self, model_name: str, inner_session: ort.InferenceSession):
        self.model_name = model_name
        self.inner_session = inner_session

    def normalize(
        self,
        img: PILImage,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        print('base sssion 6')

        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / np.max(im_ary)

        print('base sssion 7')

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        print('base sssion 8')


        tmpImg = tmpImg.transpose((2, 0, 1))

        print('base sssion 9')

        result = {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmpImg, 0)
            .astype(np.float32)
        }
        print('base sssion 10')
        return result

    def predict(self, img: PILImage) -> List[PILImage]:
        raise NotImplementedError
