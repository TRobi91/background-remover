from enum import Enum

import click
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import _version
import bg
import session_factory
import session_base

@click.group()
@click.version_option(version=_version.get_versions()["version"])
def main() -> None:
    pass

app = FastAPI(
        title="Rembg",
        description="Rembg is a tool to remove images background. That is it.",
        version=_version.get_versions()["version"],
        contact={
            "name": "Daniel Gatis",
            "url": "https://github.com/danielgatis",
            "email": "danielgatis@gmail.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://github.com/danielgatis/rembg/blob/main/LICENSE.txt",
        }
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, session_base.BaseSession] = {}

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
        bg.remove(
            content,
            session=sessions.setdefault(
                commons.model.value, session_factory.new_session(commons.model.value)
            ),
            alpha_matting=commons.a,
            alpha_matting_foreground_threshold=commons.af,
            alpha_matting_background_threshold=commons.ab,
            alpha_matting_erode_size=commons.ae,
            only_mask=commons.om,
            post_process_mask=commons.ppm,
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

