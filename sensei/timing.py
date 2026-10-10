"""Run statistics for the Sensei pipeline.

Collects per-area and per-image wall times plus backend token-level timings
(Ollama prefill/decode durations), and formats a summary. The summary feeds
directly into the pitch's "cost per analysed image" criterion.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


def percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile of an ascending-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[min(max(idx, 0), len(sorted_vals) - 1)]


@dataclass
class AreaStat:
    camera: str
    image: str
    area_id: str
    wall_s: float
    samples_ok: int
    samples_total: int
    cached_calls: int = 0
    prompt_tokens: int = 0
    decode_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0


@dataclass
class RunStats:
    started: float = field(default_factory=time.perf_counter)
    areas: list[AreaStat] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)

    def record_area(self, **kwargs: object) -> None:
        self.areas.append(AreaStat(**kwargs))  # type: ignore[arg-type]

    def record_image(self, camera: str, image: str, wall_s: float) -> None:
        self.images.append(
            {"camera": camera, "image": image, "wall_s": round(wall_s, 3)}
        )

    @property
    def total_s(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> str:
        out = ["", "=" * 48, " RUN TIMING SUMMARY", "=" * 48]
        out.append(f"Total wall time:   {self.total_s:.1f} s")
        if self.images:
            vals = [img["wall_s"] for img in self.images]
            out.append(
                f"Per image (n={len(vals)}): mean {sum(vals) / len(vals):.1f} s"
            )
        if self.areas:
            vals = sorted(a.wall_s for a in self.areas)
            out.append(
                "Per area  (n={n}): mean {mean:.2f} s | median {med:.2f} s | p95 {p95:.2f} s".format(
                    n=len(vals),
                    mean=sum(vals) / len(vals),
                    med=percentile(vals, 50),
                    p95=percentile(vals, 95),
                )
            )
        calls = sum(a.samples_total for a in self.areas)
        cached = sum(a.cached_calls for a in self.areas)
        if calls:
            out.append(f"VLM calls:       {calls} ({cached} served from cache)")
        fresh = [a for a in self.areas if a.prefill_ms > 0 or a.decode_ms > 0]
        if fresh:
            n = len(fresh)
            out.append(
                "Prefill/call:    {pm:.0f} ms mean ({pt:.0f} tok mean)".format(
                    pm=sum(a.prefill_ms for a in fresh) / n,
                    pt=sum(a.prompt_tokens for a in fresh) / n,
                )
            )
            out.append(
                "Decode/call:     {dm:.0f} ms mean ({dt:.1f} tok mean)".format(
                    dm=sum(a.decode_ms for a in fresh) / n,
                    dt=sum(a.decode_tokens for a in fresh) / n,
                )
            )
        out.append("=" * 48)
        return "\n".join(out)

    def as_dict(self) -> dict:
        return {
            "total_s": round(self.total_s, 3),
            "images": self.images,
            "areas": [asdict(a) for a in self.areas],
        }
