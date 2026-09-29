import base64
import io

import pytest
from aiohttp import web
from PIL import Image
from test_http import AUTH, configured
from test_scheduler import FakeLifecycle

from avifors.image_profiles import validate_profile
from avifors.server import create_app


def png(size):
    output = io.BytesIO()
    Image.new("RGB", size, (20, 80, 140)).save(output, format="PNG")
    return output.getvalue()


async def test_flux_profile_renders_large_and_preserves_public_size(aiohttp_client, aiohttp_server, tmp_path):
    observed = {}

    async def worker(request):
        observed.update(await request.json())
        return web.json_response({"images": [base64.b64encode(png((1024, 1024))).decode()]})

    app = web.Application()
    app.router.add_post("/sdapi/v1/txt2img", worker)
    upstream = await aiohttp_server(app)
    cfg = configured(tmp_path, upstream.make_url("/"))
    cfg.models["image"].parameters = {
        "steps": 4,
        "cfg_scale": 1,
        "sampler_name": "euler",
        "scheduler": "flux2",
        "render_sizes": {"640x640": "1024x1024"},
        "negative_prompt_mode": "instruction",
    }
    client = await aiohttp_client(create_app(cfg, FakeLifecycle()))
    response = await client.post(
        "/v1/images/generations",
        headers=AUTH,
        json={"model": "image", "prompt": "A lake", "negative_prompt": "lettering"},
    )
    assert response.status == 200
    result = await response.json()
    name = result["data"][0]["url"].rsplit("/", 1)[1]
    data = await (await client.get("/generated/" + name, headers=AUTH)).read()
    assert Image.open(io.BytesIO(data)).size == (640, 640)
    assert observed == {
        "prompt": "A lake\n\nExclude the following from the image: lettering",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "cfg_scale": 1,
        "sampler_name": "euler",
        "scheduler": "flux2",
        "batch_size": 1,
    }


async def test_sizes_are_model_specific(aiohttp_client, aiohttp_server, tmp_path):
    async def worker(request):
        return web.json_response({"images": [base64.b64encode(png((1024, 1024))).decode()]})

    app = web.Application()
    app.router.add_post("/sdapi/v1/txt2img", worker)
    upstream = await aiohttp_server(app)
    cfg = configured(tmp_path, upstream.make_url("/"))
    cfg.models["image"].parameters = {"sizes": ["1024x1024"], "default_size": "1024x1024"}
    client = await aiohttp_client(create_app(cfg, FakeLifecycle()))
    payload = {"model": "image", "prompt": "A lake"}
    assert (await client.post("/v1/images/generations", headers=AUTH, json=payload)).status == 200
    assert (
        await client.post("/v1/images/generations", headers=AUTH, json=payload | {"size": "640x640"})
    ).status == 400


@pytest.mark.parametrize(
    "parameters",
    [
        {"sizes": ["9999x9999"]},
        {"sizes": ["1024x1024"]},
        {"render_sizes": {"640x640": "1024x768"}},
        {"render_sizes": {"640x640": "512x512"}},
        {"negative_prompt_mode": "silently_ignore"},
        {"sizes": ["640x640", "640x640"]},
    ],
)
def test_invalid_image_profile_rejected(parameters):
    with pytest.raises(ValueError):
        validate_profile(parameters)


def test_render_dimensions_are_verified():
    from avifors.image_profiles import resize_png

    with pytest.raises(ValueError, match="dimensions"):
        resize_png(png((512, 512)), (640, 640), (1024, 1024))
