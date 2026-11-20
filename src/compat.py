"""Vá tương thích: TRDG 1.8.0 gọi FreeTypeFont.getsize(), hàm này đã bị xóa từ Pillow 10.
Import module này TRƯỚC khi import trdg."""
from PIL import ImageFont


def apply() -> None:
    if not hasattr(ImageFont.FreeTypeFont, "getsize"):
        def getsize(self, text, *args, **kwargs):
            _l, _t, r, b = self.getbbox(text, *args, **kwargs)
            return (r, b)
        ImageFont.FreeTypeFont.getsize = getsize


apply()
