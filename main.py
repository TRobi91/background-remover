from enum import Enum

import click
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from rembg.bg import remove
from rembg.session_factory import new_session
from rembg.session_simple import SimpleSession


@click.group()
def main() -> None:
    pass

app = FastAPI(
        title="Remove background",
        description="Remove background for photo editor",
        # contact={
        #     "name": "Daniel Gatis",
        #     "url": "https://github.com/danielgatis",
        #     "email": "danielgatis@gmail.com",
        # },
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, SimpleSession] = {}

class ModelType(str, Enum):
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"
    u2net_cloth_seg = "u2net_cloth_seg"

class CommonQueryParams:
    def __init__(
        self,
        model: ModelType = Query(
            default=ModelType.u2net,
            description="Model to use when processing image",
        ),
        a: bool = Query(default=False, description="Enable Alpha Matting"),
        af: int = Query(
            default=240,
            ge=0,
            le=255,
            description="Alpha Matting (Foreground Threshold)",
        ),
        ab: int = Query(
            default=10,
            ge=0,
            le=255,
            description="Alpha Matting (Background Threshold)",
        ),
        ae: int = Query(
            default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
        ),
        om: bool = Query(default=False, description="Only Mask"),
        ppm: bool = Query(default=False, description="Post Process Mask"),
    ):
        self.model = model
        self.a = a
        self.af = af
        self.ab = ab
        self.ae = ae
        self.om = om
        self.ppm = ppm

class CommonQueryPostParams:
    def __init__(
        self,
        model: ModelType = Form(
            default=ModelType.u2net,
            description="Model to use when processing image",
        ),
        a: bool = Form(default=False, description="Enable Alpha Matting"),
        af: int = Form(
            default=240,
            ge=0,
            le=255,
            description="Alpha Matting (Foreground Threshold)",
        ),
        ab: int = Form(
            default=10,
            ge=0,
            le=255,
            description="Alpha Matting (Background Threshold)",
        ),
        ae: int = Form(
            default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
        ),
        om: bool = Form(default=False, description="Only Mask"),
        ppm: bool = Form(default=False, description="Post Process Mask"),
    ):
        self.model = model
        self.a = a
        self.af = af
        self.ab = ab
        self.ae = ae
        self.om = om
        self.ppm = ppm


def im_without_bg(content: bytes, commons: CommonQueryParams) -> Response:
    return Response(
        remove(
            content,
            sessions.setdefault(
                commons.model.value, new_session(commons.model.value)
            ),
        ),
        media_type="image/png",
    )


@app.post(
    path="/",
    tags=["Background Removal"],
    summary="Remove from Stream",
    description="Removes the background from an image sent within the request itself.",
)
async def post_index(
    file: bytes = File(
        default=...,
        description="Image file (byte stream) that has to be processed.",
    ),
    commons: CommonQueryPostParams = Depends(),
):
    return await asyncify(im_without_bg)(file, commons)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=6000, log_level="debug")
