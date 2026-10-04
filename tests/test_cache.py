import pytest

from jev_rag_bench.cache.local_cache import LocalCache


def test_cache_miss_then_hit(tmp_path):
    cache = LocalCache(tmp_path)
    assert cache.get("jev", "jev-1.13.0", "v1", "123") is None
    cache.put("jev", "jev-1.13.0", "v1", "123", {"noul": 0.9})
    assert cache.get("jev", "jev-1.13.0", "v1", "123") == {"noul": 0.9}


def test_cache_rejects_path_traversal(tmp_path):
    cache = LocalCache(tmp_path)
    with pytest.raises(ValueError):
        cache.get("jev", "../../etc", "v1", "123")
