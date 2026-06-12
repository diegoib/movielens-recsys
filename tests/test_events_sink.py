"""Tests for pure functions in src/data/events_sink.py."""

from __future__ import annotations

from src.data.events_sink import _parquet_path


class TestParquetPath:
    def test_path_includes_date_partition(self) -> None:
        path = _parquet_path("/tmp/events", 0)
        assert "/dt=" in path
        assert "part-0000.parquet" in path

    def test_path_increments_part_number(self) -> None:
        path0 = _parquet_path("/tmp/events", 0)
        path1 = _parquet_path("/tmp/events", 1)
        assert "part-0000" in path0
        assert "part-0001" in path1

    def test_gcs_path_preserved(self) -> None:
        path = _parquet_path("gs://my-bucket/events", 5)
        assert path.startswith("gs://my-bucket/events/dt=")
        assert "part-0005.parquet" in path
