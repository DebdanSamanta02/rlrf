"""
rlrf/rendering/renderer.py
==========================
CairoSVG-based SVG rasterizer with paper guardrails (§5, App. C.3).

Guardrails implemented (paper §5 "Constraints & Guardrails"):
    1. Force render at reference image size regardless of viewBox attributes,
       preventing the model from "hacking" the reward by scaling the viewBox.
    2. Strip <text> elements before rendering (for Text2SVG; disabled for Im2SVG)
       to prevent the model from "cheating" by simply writing the text prompt.
    3. Handle rendering failures gracefully — return a blank image so the
       reward function assigns minimum reward rather than crashing.

Usage:
    renderer = SVGRenderer(size=224, strip_text=False)
    img_array = renderer.render(svg_string)   # np.ndarray [H, W, 3] uint8
    pil_img   = renderer.render_pil(svg_string)  # PIL.Image
"""

from __future__ import annotations

import io
import re
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cairosvg
    _CAIROSVG_AVAILABLE = True
except ImportError:
    _CAIROSVG_AVAILABLE = False
    logger.warning(
        "cairosvg not installed. SVG rendering will return blank images. "
        "Install with: pip install cairosvg"
    )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _strip_text_elements(svg_string: str) -> str:
    """Remove <text>, <tspan>, and <image> nodes (guardrails, §5).
    
    Stripping <image> is critical to prevent the model from "reward hacking"
    by embedding base64 raster images instead of drawing vectors!
    """
    svg_string = re.sub(r"<text[^>]*>.*?</text>", "", svg_string,
                        flags=re.DOTALL | re.IGNORECASE)
    svg_string = re.sub(r"<tspan[^>]*>.*?</tspan>", "", svg_string,
                        flags=re.DOTALL | re.IGNORECASE)
    # Strip <image ... /> or <image ...>...</image> tags
    svg_string = re.sub(r"<image[^>]*>.*?</image>", "", svg_string,
                        flags=re.DOTALL | re.IGNORECASE)
    svg_string = re.sub(r"<image[^>]*/>", "", svg_string,
                        flags=re.DOTALL | re.IGNORECASE)
    return svg_string


def _force_viewbox(svg_string: str, width: int, height: int) -> str:
    """Override the SVG viewBox / width / height to a fixed render size.

    Paper §5: "Enforce rendering at the reference image size regardless of
    predicted viewBox attributes."  Without this, the model can hack the
    L2 reward by emitting a tiny or empty viewBox.
    """
    pat = re.compile(r"(<svg)([^>]*)(>)", re.IGNORECASE | re.DOTALL)
    match = pat.search(svg_string)
    if not match:
        return svg_string

    attrs = match.group(2)
    attrs = re.sub(r'\s+width\s*=\s*["\'][^"\']*["\']', "", attrs)
    attrs = re.sub(r'\s+height\s*=\s*["\'][^"\']*["\']', "", attrs)
    attrs = re.sub(r'\s+viewBox\s*=\s*["\'][^"\']*["\']', "", attrs)

    new_attrs = (
        f'{attrs} width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"'
    )
    new_tag = f"{match.group(1)}{new_attrs}{match.group(3)}"
    return svg_string[: match.start()] + new_tag + svg_string[match.end():]


def _blank_image(size: int) -> np.ndarray:
    """White H×W×3 uint8 fallback returned on rendering failure."""
    return np.full((size, size, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# SVGRenderer
# ---------------------------------------------------------------------------

class SVGRenderer:
    """Rasterize SVG strings to fixed-size RGB numpy arrays via CairoSVG.

    Args:
        size (int): Output image side length in pixels (paper: 224).
        strip_text (bool): Strip <text> elements before rendering.
            Set True for Text2SVG to prevent reward hacking (paper §5).
        enforce_viewbox (bool): Force fixed viewBox (paper §5). Default True.
    """

    def __init__(
        self,
        size: int = 224,
        strip_text: bool = False,
        enforce_viewbox: bool = True,
    ) -> None:
        self.size = size
        self.strip_text = strip_text
        self.enforce_viewbox = enforce_viewbox

        if not _CAIROSVG_AVAILABLE:
            logger.warning("SVGRenderer: cairosvg unavailable; renders will be blank.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, svg_string: str) -> np.ndarray:
        """Render SVG string → RGB numpy array [H, W, 3] uint8.

        Returns a white blank image on any rendering failure so the
        reward function receives minimum reward rather than crashing.
        """
        svg_string = self._preprocess(svg_string)
        if not svg_string or not _CAIROSVG_AVAILABLE:
            return _blank_image(self.size)

        try:
            png_bytes = cairosvg.svg2png(
                bytestring=svg_string.encode("utf-8"),
                output_width=self.size,
                output_height=self.size,
            )
            # Composite onto white background to avoid transparent -> black conversion
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            bg = Image.new("RGBA", img.size, "WHITE")
            bg.alpha_composite(img)
            return np.array(bg.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            logger.debug("SVG render failed (%s). Returning blank.", exc)
            return _blank_image(self.size)

    def render_pil(self, svg_string: str) -> Image.Image:
        """Render SVG → PIL Image (RGB). Convenience wrapper."""
        return Image.fromarray(self.render(svg_string))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, svg_string: str) -> str:
        """Clean model output and apply guardrails."""
        if not svg_string:
            return ""
        svg_string = self._strip_markdown_fences(svg_string)
        svg_string = self._extract_svg(svg_string)
        if not svg_string:
            return ""
        if self.strip_text:
            svg_string = _strip_text_elements(svg_string)
        if self.enforce_viewbox:
            svg_string = _force_viewbox(svg_string, self.size, self.size)
        return svg_string

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove ```svg ... ``` code fences the model may prepend."""
        text = re.sub(r"```(?:svg|xml)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        return text.strip()

    @staticmethod
    def _extract_svg(text: str) -> str:
        """Extract the first <svg>...</svg> block from arbitrary text."""
        match = re.search(r"(<svg[\s\S]*?</svg>)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        if text.strip().lower().startswith("<svg"):
            return text.strip()
        return ""
