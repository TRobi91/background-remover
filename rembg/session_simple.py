from typing import List

print('session simple 1')
import numpy as np
print('session simple 2')

from PIL import Image
print('session simple 3')

from PIL.Image import Image as PILImage
print('session simple 4')

from session_base import BaseSession
print('session simple 5')

class SimpleSession(BaseSession):
    def predict(self, img: PILImage) -> List[PILImage]:
        print('session simple 6')

        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)
            ),
        )

        print('session simple 7')


        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        print('session simple 8')


        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        print('session simple 9')


        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")

        print('session simple 10')

        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        print('session simple 11')


        return [mask]
