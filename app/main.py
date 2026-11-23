"""FastAPI handoff service for gauge-reading packages.

Endpoint summary
----------------
POST /api/packages   CSV + limits in, self-describing binary package out.
POST /api/inspect    Package (and optionally the original CSV) in, decoded
                     readings, verdicts and an integrity report out.
GET  /healthz        liveness probe.

Interactive docs (upload forms included) are served at /docs.
"""

from __future__ import annotations

import itertools
import json

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse

from . import pack, transfer

app = FastAPI(
    title="轨距读数交接 API",
    version="1.0.0",
    description=(
        "按采样顺序接收轨距 CSV（整数微米），生成自带解码信息与合格区间的"
        "二进制包。无损包超过字节上限时，允许在绝对误差限内近似；仍装不下则"
        "返回 413，绝不交付半包。"
    ),
)

PACKAGE_MEDIA_TYPE = "application/octet-stream"
PACKAGE_FILENAME = "gauge-package.tgp"
MAX_PACKAGE_BYTES = 256 * 1024 * 1024


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/healthz", summary="Health check")
def healthz() -> dict:
    return {"status": "ok"}


@app.exception_handler(transfer.InputError)
async def _input_error_handler(_request, exc: transfer.InputError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(pack.PackageTooLarge)
async def _too_large_handler(_request, exc: pack.PackageTooLarge):
    return JSONResponse(status_code=413, content={"detail": str(exc)})


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise transfer.InputError(f"upload exceeds {limit} bytes")
    return data


@app.post(
    "/api/packages",
    summary="CSV → 交接包",
    response_class=Response,
    responses={
        200: {
            "content": {PACKAGE_MEDIA_TYPE: {}},
            "description": (
                "自描述交接包。响应头 X-Package-Samples / X-Package-Bytes / "
                "X-Package-Codec / X-Package-Approximated / "
                "X-Package-Max-Error 概括包内容。"
            ),
        },
        400: {"description": "CSV 或参数不合法"},
        413: {"description": "在字节上限内无法交付完整包"},
    },
)
async def create_package(
    file: UploadFile = File(..., description="按采样顺序排列的整数微米 CSV"),
    lower: int = Form(..., description="合格闭区间下界（微米，含）"),
    upper: int = Form(..., description="合格闭区间上界（微米，含）"),
    eps: int = Form(..., description="逐点绝对误差上限（微米，0 表示必须无损）"),
    byte_limit: int = Form(..., description="包的字节上限（含头部）"),
):
    raw = await _read_upload(file, transfer.MAX_CSV_BYTES)
    transfer.check_params(lower, upper, eps, byte_limit)
    readings = transfer.parse_readings(raw)

    # deliver() only returns a package that has already been decoded and
    # checked against the originals; a failure raises before any bytes leave.
    delivery = transfer.deliver(readings.values, lower, upper, eps, byte_limit)

    headers = {
        "Content-Disposition": f'attachment; filename="{PACKAGE_FILENAME}"',
        "X-Package-Samples": str(delivery.original_len),
        "X-Package-Bytes": str(len(delivery.package)),
        "X-Package-Codec": delivery.codec,
        "X-Package-Approximated": "true" if delivery.approximated else "false",
        "X-Package-Max-Error": str(delivery.max_error),
        "X-Package-Error-Limit": str(eps),
        "X-Package-Verified": "true" if delivery.verified else "false",
    }
    return Response(
        content=delivery.package,
        media_type=PACKAGE_MEDIA_TYPE,
        headers=headers,
    )


@app.post(
    "/api/inspect",
    summary="交接包 → 读数与判定",
    responses={
        200: {"description": "按原采样顺序解码出的数值、判定与完整性报告"},
        400: {"description": "包损坏、格式不支持或与原值核对不一致"},
    },
)
async def inspect_package(
    file: UploadFile = File(..., description="交接包（.tgp）"),
    original: UploadFile | None = File(
        None,
        description="可选：原始 CSV。提供时逐点核对误差上限与判定一致性",
    ),
):
    try:
        blob = await _read_upload(file, MAX_PACKAGE_BYTES)
    finally:
        # Drop the multipart spool file as soon as the bytes are read: on a
        # memory-capped container /tmp is tmpfs and counts toward the limit.
        await file.close()
    try:
        # Decoded exactly once; the report below and the point-by-point check
        # both work on this single batch of values.  Values stay in a compact
        # int32 array so a 2M-sample package costs ~8 MiB, not ~80.
        decoded = pack.decode_package_array(blob)
    except (ValueError, pack.PackError) as exc:
        raise transfer.InputError(f"package failed to decode: {exc}") from exc
    del blob  # the compact array is all the report needs from here on

    summary = None
    if original is not None:
        try:
            expected = await transfer.read_readings_array(original)
        finally:
            await original.close()  # release this run's upload staging
        try:
            # The whole series is checked before any response bytes leave:
            # a violation at the very last sample still returns 400.
            summary = transfer.verify_decoded_values(decoded, expected)
        finally:
            del expected  # release this run's staging even on a failed check

    return StreamingResponse(
        _report_chunks(decoded, summary),
        media_type="application/json",
    )


_JSON_TRUE = "true"
_JSON_FALSE = "false"

# Pre-quoted JSON forms of the fixed verdict vocabulary.
_VERDICT_JSON = {"narrow": '"narrow"', "pass": '"pass"', "wide": '"wide"'}
_VERDICT_LABEL_JSON = {"narrow": '"偏窄"', "pass": '"合格"', "wide": '"偏宽"'}

# Report arrays are serialised in slices of this many samples, so the JSON
# body is produced incrementally instead of as one giant string.
_REPORT_SLICE = 100_000


def _joined_slices(items):
    """Yield comma-joined slices of *items*, keeping array commas balanced."""
    iterator = iter(items)
    separator = ""
    while True:
        slice_ = list(itertools.islice(iterator, _REPORT_SLICE))
        if not slice_:
            return
        yield separator + ",".join(slice_)
        separator = ","


def _report_chunks(decoded: dict, summary: dict | None):
    """Stream the inspect report as JSON fragments, in sampling order.

    Field set and order match the historical JSONResponse exactly; the body is
    just never materialised as one string (2M samples ≈ 40 MB of JSON).
    """
    values = decoded["values"]
    lower, upper = decoded["lower"], decoded["upper"]
    verdict = pack.verdict
    yield (
        f'{{"count":{decoded["count"]},"bytes":{decoded["bytes"]},'
        f'"codec":{json.dumps(decoded["codec"], ensure_ascii=False)},'
        f'"approximated":'
        f'{_JSON_TRUE if decoded["approximated"] else _JSON_FALSE},'
        f'"pass_interval":{{"lower":{lower},"upper":{upper}}},'
        f'"error_limit":{decoded["eps"]},"values":['
    )
    yield from _joined_slices(str(value) for value in values)
    yield '],"verdicts":['
    yield from _joined_slices(
        _VERDICT_JSON[verdict(value, lower, upper)] for value in values
    )
    yield '],"verdict_labels":['
    yield from _joined_slices(
        _VERDICT_LABEL_JSON[verdict(value, lower, upper)] for value in values
    )
    tail = (
        f'],"checked_against_original":'
        f'{_JSON_TRUE if summary is not None else _JSON_FALSE}'
    )
    if summary is not None:
        tail += f',"max_error":{summary["max_error"]}'
    yield tail + "}"
