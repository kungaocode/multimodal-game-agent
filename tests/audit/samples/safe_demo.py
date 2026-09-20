"""Safe demo target for performance tests."""


def heavy_function(n: int = 1000) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total
