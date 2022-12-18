import io
from enum import Enum
from typing import List, Optional, Union
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from session_simple import SimpleSession


class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2


def get_concat_v_multi(imgs: List[PILImage]) -> PILImage:
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot


def get_concat_v(img1: PILImage, img2: PILImage) -> PILImage:
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


def remove(
    data: Union[bytes, PILImage, np.ndarray],
    session: Optional[SimpleSession] = None,
) -> Union[bytes, PILImage, np.ndarray]:

    if isinstance(data, PILImage):
        return_type = ReturnType.PILLOW
        img = data
    elif isinstance(data, bytes):
        return_type = ReturnType.BYTES
        img = Image.open(io.BytesIO(data))
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = Image.fromarray(data)
    else:
        raise ValueError("Input type {} is not supported.".format(type(data)))

    masks = session.predict(img)
    cutouts = []

    for mask in masks:
        cutouts.append(mask)

    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    if ReturnType.PILLOW == return_type:
        return cutout

    if ReturnType.NDARRAY == return_type:
        return np.asarray(cutout)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()
