"""CSV ingest for ordered gauge readings and the package/verify pipeline."""

from __future__ import annotations

import codecs
import csv
import io
import re
import struct
import time
from dataclasses import dataclass
from typing import List, Sequence

from . import pack

MAX_CSV_BYTES = 64 * 1024 * 1024

# Uploads are parsed in chunks of this size so a multi-megabyte CSV never has
# to sit in memory next to its decoded values.
_CSV_CHUNK_BYTES = 1 << 20

# Cells are delimited by commas or newlines; blank cells are skipped, exactly
# like the row/column layouts parse_readings accepts.
_CSV_SPLIT = re.compile(r"[,\n]")

_HEADER_TOKENS = {"value", "gauge", "reading", "轨距"}


class InputError(ValueError):
    """User-facing validation problem with the uploaded CSV or parameters."""


@dataclass
class Readings:
    values: List[int]

    def __len__(self) -> int:
        return len(self.values)


def _parse_token(token: str, index: int) -> int:
    """One CSV cell → one micrometer reading, or :class:`InputError`."""
    try:
        number = int(token, 10)
    except ValueError:
        raise InputError(
            f"sample {index} is not an integer number of micrometers: {token!r}"
        ) from None
    if not pack.INT32_MIN <= number <= pack.INT32_MAX:
        raise InputError(
            f"sample {index} ({number}) is outside signed 32-bit range"
        )
    return number


def parse_readings(raw: bytes) -> Readings:
    """Parse a CSV of integer micrometers in sampling order.

    Accepts one integer per line, or one integer per column on a single line.
    Blank lines are ignored.  A single optional header row (``value``/``gauge``)
    is tolerated.  Values must fit in signed int32.
    """
    if len(raw) > MAX_CSV_BYTES:
        raise InputError(f"CSV exceeds {MAX_CSV_BYTES} bytes")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError("CSV must be UTF-8") from exc

    values: List[int] = []
    first_token = True
    try:
        # csv.reader is lazy: rows are consumed as they are produced, so the
        # whole CSV never exists as a list of row lists.
        for row in csv.reader(io.StringIO(text)):
            for cell in row:
                token = cell.strip()
                if not token:
                    continue
                if first_token:
                    first_token = False
                    if token.lower() in _HEADER_TOKENS:
                        continue
                values.append(_parse_token(token, len(values) + 1))
    except csv.Error as exc:
        raise InputError(f"malformed CSV: {exc}") from exc

    if not values:
        raise InputError("CSV contains no readings")
    if len(values) > pack.MAX_SAMPLES:
        raise InputError(
            f"CSV contains {len(values)} readings; the limit is "
            f"{pack.MAX_SAMPLES}"
        )
    return Readings(values)


async def read_readings_array(upload, limit: int = MAX_CSV_BYTES):
    """Stream an uploaded CSV straight into a compact int32 array.

    The upload is read in fixed-size chunks and decoded incrementally, so peak
    memory stays flat no matter how large the CSV is.  Accepts the same layouts
    as :func:`parse_readings` and raises the same :class:`InputError`\\ s.
    """
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    values = pack._int32_array()
    pending = ""
    total = 0
    first_token = True

    def absorb(token: str) -> None:
        nonlocal first_token
        token = token.strip()
        if not token:
            return
        if first_token:
            first_token = False
            if token.lower() in _HEADER_TOKENS:
                return
        index = len(values) + 1
        if index > pack.MAX_SAMPLES:
            raise InputError(
                f"CSV contains {index} readings; the limit is "
                f"{pack.MAX_SAMPLES}"
            )
        values.append(_parse_token(token, index))

    while True:
        chunk = await upload.read(_CSV_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise InputError(f"upload exceeds {limit} bytes")
        try:
            pending += decoder.decode(chunk)
        except UnicodeDecodeError:
            raise InputError("CSV must be UTF-8") from None
        *tokens, pending = _CSV_SPLIT.split(pending)
        for token in tokens:
            absorb(token)
    try:
        pending += decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        raise InputError("CSV must be UTF-8") from None
    absorb(pending)

    if not values:
        raise InputError("CSV contains no readings")
    return values


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def check_params(lower: int, upper: int, eps: int, byte_limit: int) -> None:
    if lower > upper:
        raise InputError("pass interval lower bound must not exceed upper bound")
    if not (INT64_MIN <= lower and upper <= INT64_MAX):
        raise InputError("pass interval bounds must fit signed 64-bit integers")
    if eps < 0:
        raise InputError("absolute error limit must be non-negative")
    if eps > INT64_MAX:
        raise InputError("absolute error limit must fit a signed 64-bit integer")
    if byte_limit < pack.HEADER_SIZE:
        raise InputError(
            f"byte limit must be at least the {pack.HEADER_SIZE}-byte header"
        )


@dataclass
class Delivery:
    package: bytes
    approximated: bool
    original_len: int
    max_error: int
    verified: bool
    codec: str


def _compare(decoded_values, expected, lower: int, upper: int, eps: int,
             approximated: bool) -> int:
    """Check decoded values against the originals, one pass, no copies.

    Raises InputError at the first violated invariant — per-sample error bound
    or verdict agreement — and confirms the approximation flag matches the
    actual differences.  Returns the maximum absolute error seen.
    """
    verdict = pack.verdict
    max_error = 0
    differs = False
    for index, (original, restored) in enumerate(
        zip(expected, decoded_values), start=1
    ):
        error = restored - original
        if error < 0:
            error = -error
        if error > eps:
            raise InputError(
                f"sample {index} error {error} um exceeds limit {eps} um"
            )
        if error > max_error:
            max_error = error
        if error:
            differs = True
        expected_verdict = verdict(original, lower, upper)
        got_verdict = verdict(restored, lower, upper)
        if expected_verdict != got_verdict:
            raise InputError(
                f"sample {index} verdict changed after decode "
                f"({expected_verdict} -> {got_verdict})"
            )
    if approximated != differs:
        raise InputError("package approximation flag disagrees with its contents")
    return max_error


def verify_decoded_values(decoded: dict, expected: Sequence[int]) -> dict:
    """Check already-decoded package values against the original readings.

    Used by /api/inspect: the package is decoded exactly once and both the
    report and this point-by-point check work on that same batch of values.
    Raises InputError on any mismatch; returns a summary dict.
    """
    values = decoded["values"]
    if len(values) != len(expected):
        raise InputError("decoded sample count does not match input")
    max_error = _compare(
        values, expected,
        decoded["lower"], decoded["upper"], decoded["eps"],
        decoded["approximated"],
    )
    return {
        "max_error": max_error,
        "approximated": decoded["approximated"],
        "codec": decoded["codec"],
    }


def verify_package(blob: bytes, expected: Sequence[int], lower: int, upper: int,
                   eps: int) -> dict:
    """Decode and check a package against the originals.

    Raises InputError if any invariant is violated: count, per-sample error
    bound, exact verdict match against the original values, or the embedded
    pass interval.  Returns a summary dict.
    """
    try:
        decoded = pack.decode_package_array(blob)
    except (ValueError, pack.PackError) as exc:
        raise InputError(f"package failed to decode: {exc}") from exc

    if len(decoded["values"]) != len(expected):
        raise InputError("decoded sample count does not match input")
    if (decoded["lower"], decoded["upper"]) != (lower, upper):
        raise InputError("package pass interval does not match the request")
    if decoded["eps"] != eps:
        raise InputError("package error limit does not match the request")

    max_error = _compare(
        decoded["values"], expected, lower, upper, eps,
        decoded["approximated"],
    )
    return {
        "max_error": max_error,
        "approximated": decoded["approximated"],
        "codec": decoded["codec"],
    }


def _legal_band(
    values: Sequence[int], lower: int, upper: int, eps: int
) -> List[tuple[int, int]]:
    """Per-sample interval of replacements that stay within *eps* and keep the
    verdict unchanged.  A sample pinned on an interval edge therefore gets a
    one-point band and simply cannot be approximated.  Bands are intersected
    with the signed 32-bit range so every candidate stays representable."""
    bands = []
    for v in values:
        lo, hi = v - eps, v + eps
        if v < lower:            # narrow: must stay strictly below
            hi = min(hi, lower - 1)
        elif v > upper:          # wide: must stay strictly above
            lo = max(lo, upper + 1)
        else:                    # pass: must stay inside the closed interval
            lo = max(lo, lower)
            hi = min(hi, upper)
        bands.append((max(lo, pack.INT32_MIN), min(hi, pack.INT32_MAX)))
    return bands


def _approximation_candidates(
    values: Sequence[int], lower: int, upper: int, eps: int
):
    """Generate bounded-error approximations, coarsest first.

    Each replacement is clamped into the per-sample legal band, so every
    candidate keeps the error within +/- eps and every verdict identical.
    :func:`verify_package` independently re-checks the finished package.

    At ``eps == 0`` only error-free rewrites are possible in principle, but
    quantization/run-averaging can still be lossless for some series, so they
    are offered too (and flagged as approximated when they win).
    """
    n = len(values)
    if n == 0:
        return
    bands = _legal_band(values, lower, upper, eps)

    def snap(i: int, target: int) -> int:
        lo, hi = bands[i]
        return max(lo, min(hi, target))

    # Candidate plan scales with series length so a large upload cannot turn
    # the fallback search into minutes of CPU.
    if n > 1_000_000:
        steps = (4096, 1024, 256, 64, 16, 4)
        blocks = (64, 16, 4, 2)
        smooth_rounds = 1
    elif n > 200_000:
        steps = (4096, 2048, 1024, 256, 64, 16, 4, 2)
        blocks = (64, 48, 16, 8, 4, 3, 2)
        smooth_rounds = 2
    else:
        steps = (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2)
        blocks = (64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2)
        smooth_rounds = 4

    # 1. Quantisation: snap to a coarser grid inside the error band.  This is
    #    what actually removes entropy when the readings carry low-order noise.
    #    Each step yields the plain snap plus a run-repeated variant, which
    #    removes residual jitter after quantisation.
    for step in steps:
        if step // 2 > eps:
            continue
        quantised = [
            snap(i, round(v / step) * step) for i, v in enumerate(values)
        ]
        yield quantised
        yield _run_repeat(quantised, bands)

    # 2. Run averaging: repeating one value over each block removes detail the
    #    error budget allows us to drop, and compresses extremely well.  A block
    #    is only held when one value is legal for every sample in it.
    for block in blocks:
        yield _run_repeat(values, bands, block)

    # 3. Clamped low-pass smoothing for drift-like series.
    base = list(values)
    for _ in range(smooth_rounds):
        base = [
            snap(i, round((base[max(0, i - 1)] + base[i] + base[min(n - 1, i + 1)]) / 3))
            for i in range(n)
        ]
        yield list(base)


def _run_repeat(
    values: Sequence[int], bands: Sequence[tuple[int, int]], block: int = 2
) -> List[int]:
    """Hold one legal value across each run of *block* samples.

    Only samples whose legal band (bounded error and unchanged verdict) shares
    a common value take part; the rest are copied verbatim.  With ``block == 2``
    it also merges quantised plateaus, which compresses markedly better.
    """
    out: List[int] = []
    n = len(values)
    for start in range(0, n, block):
        window = range(start, min(n, start + block))
        lo = max(bands[i][0] for i in window)
        hi = min(bands[i][1] for i in window)
        if lo <= hi:
            held = max(lo, min(hi, round(
                sum(values[i] for i in window) / len(window))))
            out.extend([held] * len(window))
        else:
            out.extend(values[i] for i in window)
    return out


def _encode_candidates(
    data: Sequence[int],
    flags: int,
    lower: int,
    upper: int,
    eps: int,
    *,
    allow_deflate_raw: bool = True,
):
    """Yield (size, codec, blob) for one sample series, cheap codecs first.

    ``deflate-int32`` is the most expensive encoder and only wins on data whose
    deltas are unhelpful, so callers may turn it off for very long series.
    A codec that cannot represent this data (for example a raw int32 stream
    with an out-of-range endpoint) is skipped rather than failing the request.
    The generator stops as soon as the caller breaks.
    """
    n = len(data)
    order = [pack.CODEC_DELTA_VARINT, pack.CODEC_DELTA2_VARINT]
    if n <= 4096:
        order.append(pack.CODEC_RAW)
    order.append(pack.CODEC_DEFLATE_DELTA)
    if allow_deflate_raw:
        order.append(pack.CODEC_DEFLATE_RAW)
    for codec in order:
        try:
            payload = pack.encode_payload(codec, data)
            blob = pack.build_package(
                codec, flags, data, payload, lower, upper, eps
            )
        except (pack.PackError, struct.error, OverflowError):
            continue
        yield len(blob), codec, blob


def _best_under(
    data: Sequence[int],
    flags: int,
    lower: int,
    upper: int,
    eps: int,
    byte_limit: int,
    *,
    allow_deflate_raw: bool = True,
    deadline: float | None = None,
) -> tuple[bytes | None, int]:
    """Smallest encoding of *data* that fits *byte_limit*, plus the floor size.

    If *deadline* (a ``time.monotonic`` value) passes, encoding stops early and
    whatever was already measured is returned; a package that already fits is
    therefore never lost to the clock.
    """
    best = None
    floor = None
    for size, _codec, blob in _encode_candidates(
        data, flags, lower, upper, eps, allow_deflate_raw=allow_deflate_raw
    ):
        floor = size if floor is None else min(floor, size)
        if size <= byte_limit and (best is None or size < len(best)):
            best = blob
        if deadline is not None and time.monotonic() > deadline:
            break
    return best, floor


# Wall-clock budget for the bounded-error search.  Requests that need more
# than this are reported as failures rather than left hanging; the lossless
# attempt and the final verify always run outside of it.
APPROX_TIME_BUDGET = 20.0


def deliver(
    values: Sequence[int],
    lower: int,
    upper: int,
    eps: int,
    byte_limit: int,
) -> Delivery:
    """Produce a verified package within *byte_limit*.

    Lossless encodings are tried first and the smallest wins.  Only when none
    fit does the bounded-error fallback run: candidates go from coarsest (most
    compressible) to finest, and the first one that yields a package at or
    under the limit is verified and returned.  If nothing fits, no bytes are
    delivered and :class:`PackageTooLarge` is raised.
    """
    n = len(values)
    # Long series cannot afford the exhaustive search; the coarse candidates
    # that matter most are still all tried and the clock bounds the rest.
    allow_deflate_raw = n <= 200_000

    # Fast path: lossless fits.  This pass is linear in the sample count and
    # is always allowed to finish; only the fallback search is budgeted.
    lossless, floor = _best_under(
        values, 0, lower, upper, eps, 10**18,
        allow_deflate_raw=allow_deflate_raw,
    )
    if len(lossless) <= byte_limit:
        summary = verify_package(lossless, values, lower, upper, eps)
        return Delivery(
            package=lossless,
            approximated=False,
            original_len=n,
            max_error=summary["max_error"],
            verified=True,
            codec=summary["codec"],
        )

    # Bounded-error fallback, coarsest admissible approximation first.  Each
    # candidate keeps every sample inside its own +/- eps and verdict band.
    deadline = time.monotonic() + APPROX_TIME_BUDGET
    timed_out = False
    for candidate in _approximation_candidates(values, lower, upper, eps):
        if time.monotonic() > deadline:
            timed_out = True
            break
        if list(candidate) == list(values):
            continue  # identical to the lossless data, already known not to fit
        best, candidate_floor = _best_under(
            candidate, pack.FLAG_APPROXIMATED, lower, upper, eps, byte_limit,
            allow_deflate_raw=allow_deflate_raw, deadline=deadline,
        )
        floor = min(floor, candidate_floor)
        if best is None:
            continue
        try:
            summary = verify_package(best, values, lower, upper, eps)
        except InputError:
            continue  # never deliver a package that failed its own check
        return Delivery(
            package=best,
            approximated=summary["approximated"],
            original_len=n,
            max_error=summary["max_error"],
            verified=True,
            codec=summary["codec"],
        )

    reason = (
        "search budget exhausted before a fitting layout was confirmed"
        if timed_out else
        "no complete package fits the byte limit"
    )
    raise pack.PackageTooLarge(
        f"{reason}: smallest encodable package seen is {floor} bytes "
        f"(limit {byte_limit})"
    )
