"""Tests for pure functions in src/data/build_retrain_dataset.py."""

from __future__ import annotations

import polars as pl

from src.data.build_retrain_dataset import _assign_labels, _expand_user_features


class TestAssignLabels:
    def _make_inference(self, rec_id: str, movie_id: int) -> pl.DataFrame:
        return pl.DataFrame({"recommendation_id": [rec_id], "movie_id": [movie_id]})

    def _make_clicks(self, rec_id: str, movie_id: int) -> pl.DataFrame:
        return pl.DataFrame({"recommendation_id": [rec_id], "movie_id": [movie_id]})

    def test_click_for_same_rec_and_movie_gets_label_one(self) -> None:
        inf = self._make_inference("rec-1", 42)
        clicks = self._make_clicks("rec-1", 42)
        result = _assign_labels(inf, clicks)
        assert result["label"][0] == 1

    def test_no_click_gets_label_zero(self) -> None:
        inf = self._make_inference("rec-1", 42)
        clicks = pl.DataFrame(
            {
                "recommendation_id": pl.Series([], dtype=pl.String),
                "movie_id": pl.Series([], dtype=pl.Int64),
            }
        )
        result = _assign_labels(inf, clicks)
        assert result["label"][0] == 0

    def test_click_different_movie_does_not_match(self) -> None:
        inf = self._make_inference("rec-1", 42)
        clicks = self._make_clicks("rec-1", 99)  # different movie_id
        result = _assign_labels(inf, clicks)
        assert result["label"][0] == 0

    def test_click_different_rec_id_does_not_match(self) -> None:
        inf = self._make_inference("rec-1", 42)
        clicks = self._make_clicks("rec-2", 42)  # different recommendation_id
        result = _assign_labels(inf, clicks)
        assert result["label"][0] == 0

    def test_label_column_is_int32(self) -> None:
        inf = self._make_inference("rec-1", 42)
        clicks = self._make_clicks("rec-1", 42)
        result = _assign_labels(inf, clicks)
        assert result["label"].dtype == pl.Int32


class TestExpandUserFeatures:
    def _make_df(self, user_features_json: str) -> pl.DataFrame:
        return pl.DataFrame({"user_features": [user_features_json]})

    def test_parses_n_clicks(self) -> None:
        df = self._make_df('{"n_clicks_last_7d": 5}')
        result = _expand_user_features(df)
        assert result["n_clicks_last_7d"][0] == 5

    def test_parses_genre_affinity_list(self) -> None:
        affinity = [0.1] * 19
        import json

        df = self._make_df(json.dumps({"genre_affinity_last_7d": affinity}))
        result = _expand_user_features(df)
        assert len(result["genre_affinity_last_7d"][0]) == 19

    def test_empty_features_returns_defaults(self) -> None:
        df = self._make_df("{}")
        result = _expand_user_features(df)
        assert result["n_clicks_last_7d"][0] == 0
        assert result["days_since_last_activity"][0] == 0.0

    def test_user_features_column_dropped(self) -> None:
        df = self._make_df('{"n_clicks_last_7d": 3}')
        result = _expand_user_features(df)
        assert "user_features" not in result.columns
