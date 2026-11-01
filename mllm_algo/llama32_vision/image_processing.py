"""Llama 3.2-Vision 的高分辨率图像预处理: 把一张图按最合适的宽高比切成最多 4 个 560×560 的 tile。

    1. 枚举所有 tile 数不超过 max_num_tiles 的布局 (列数 × 行数), 每种布局对应一个 aspect_ratio_id
    2. 选一个让图像缩放最合理的布局 (get_optimal_canvas)
    3. 保持宽高比缩放到画布内, 右下补 0, 再切成 tile
    4. 不足 max_num_tiles 的位置补空 tile, 用 aspect_ratio_mask 标记哪些 tile 是真实的
"""
from typing import List, Tuple

import torch
import torch.nn.functional as F


def get_supported_aspect_ratios(max_num_tiles: int = 4) -> List[Tuple[int, int]]:
    """返回所有 (列数, 行数) 布局。max_num_tiles=4 时有 8 种:
    (1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (3,1) (4,1)。aspect_ratio_id = 下标 + 1, 0 留给 padding。
    """
    return [(w, h) for w in range(1, max_num_tiles + 1) for h in range(1, max_num_tiles + 1) if w * h <= max_num_tiles]


def get_optimal_canvas(height: int, width: int, tile_size: int, max_num_tiles: int) -> Tuple[int, int]:
    """选择 tile 布局, 返回 (列数, 行数)。

    对每种布局计算 "图像保持宽高比放进画布" 需要的缩放倍数:
      - 如果有布局可以放大 (scale >= 1), 选放大倍数最小的, 避免过度插值
      - 否则只能缩小, 选缩小最少的, 尽量保留细节
    缩放倍数相同时选面积最小 (tile 最少) 的布局, 节省计算。
    """
    candidates = []
    for cols, rows in get_supported_aspect_ratios(max_num_tiles):
        scale = min(cols * tile_size / width, rows * tile_size / height)
        candidates.append((scale, cols * rows, (cols, rows)))

    upscales = [c for c in candidates if c[0] >= 1]
    target_scale = min(c[0] for c in upscales) if upscales else max(c[0] for c in candidates)
    best = min((c for c in candidates if c[0] == target_scale), key=lambda c: c[1])
    return best[2]


def tile_image(image: torch.Tensor, tile_size: int = 560, max_num_tiles: int = 4):
    """image: [3, H, W] -> (tiles [num_tiles, 3, tile, tile], aspect_ratio_id)"""
    _, H, W = image.shape
    cols, rows = get_optimal_canvas(H, W, tile_size, max_num_tiles)

    scale = min(cols * tile_size / W, rows * tile_size / H)
    new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))
    resized = F.interpolate(image[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
    canvas = image.new_zeros(3, rows * tile_size, cols * tile_size)
    canvas[:, :new_h, :new_w] = resized

    # [3, rows*t, cols*t] -> [rows*cols, 3, t, t], tile 按行优先排列
    tiles = canvas.view(3, rows, tile_size, cols, tile_size).permute(1, 3, 0, 2, 4).reshape(-1, 3, tile_size, tile_size)
    aspect_ratio_id = get_supported_aspect_ratios(max_num_tiles).index((cols, rows)) + 1
    return tiles, aspect_ratio_id


def preprocess_images(images: List[torch.Tensor], tile_size: int = 560, max_num_tiles: int = 4):
    """处理一条样本中的多张图。

    return:
        pixel_values      [N_img, max_num_tiles, 3, tile, tile]
        aspect_ratio_ids  [N_img]
        aspect_ratio_mask [N_img, max_num_tiles], 1 = 真实 tile, 0 = 补齐的空 tile
    """
    pixel_values = torch.zeros(len(images), max_num_tiles, 3, tile_size, tile_size)
    aspect_ratio_ids = torch.zeros(len(images), dtype=torch.long)
    aspect_ratio_mask = torch.zeros(len(images), max_num_tiles, dtype=torch.long)
    for i, image in enumerate(images):
        tiles, ratio_id = tile_image(image, tile_size, max_num_tiles)
        pixel_values[i, :len(tiles)] = tiles
        aspect_ratio_ids[i] = ratio_id
        aspect_ratio_mask[i, :len(tiles)] = 1
    return pixel_values, aspect_ratio_ids, aspect_ratio_mask
