import math


def hilbert_curve(n):
    if n == 1:
        return [(0, 0)]

    half = n // 2
    smaller = hilbert_curve(half)

    result = []

    for x, y in smaller:
        result.append((y, x))

    for x, y in smaller:
        result.append((x, y + half))

    for x, y in smaller:
        result.append((x + half, y + half))

    for x, y in reversed(smaller):
        result.append((n - 1 - y, half - 1 - x))

    return result


def hilbert_order(height, width):
    size = max(height, width)
    size = 2 ** math.ceil(math.log2(size))

    curve = hilbert_curve(size)
    order = []

    for x, y in curve:
        if x < height and y < width:
            order.append(x * width + y)

    return order


def wind_band_hilbert(height, width, wind_angle):
    dx = math.cos(wind_angle)
    dy = math.sin(wind_angle)

    positions = []
    for i in range(height):
        for j in range(width):
            proj = i * dy + j * dx
            positions.append((proj, i, j))

    positions.sort(key=lambda x: x[0])

    num_bands = max(height, width)
    band_size = len(positions) // num_bands

    order = []
    for band_idx in range(num_bands):
        start = band_idx * band_size
        end = start + band_size if band_idx < num_bands - 1 else len(positions)
        band = positions[start:end]

        band_indices = [(p[1], p[2]) for p in band]
        band_indices.sort(key=lambda x: (x[0] + x[1]) % 2 * 1000 + x[0] * width + x[1])

        for i, j in band_indices:
            order.append(i * width + j)

    return order
