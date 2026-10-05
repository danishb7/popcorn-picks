from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from popcorn_picks import config
from popcorn_picks.model import save_model
from popcorn_picks.training import TrainingConfig, train_model


def test_streamlit_catalog_and_personalized_recommendations(
    movie_data, data_dir, tmp_path, monkeypatch
):
    model, _ = train_model(
        movie_data.ratings, TrainingConfig(min_ratings=1, n_factors=3, n_epochs=2)
    )
    path = tmp_path / "model.pkl"
    save_model(model, path)
    monkeypatch.setattr(config, "DEFAULT_DATA_DIR", data_dir)
    monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", path)
    st.cache_resource.clear()

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "app.py"))
    app.run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 5
    app.selectbox[0].set_value("Action")
    app.button[0].click().run()
    assert not app.exception
    assert "Action" in [item.value for item in app.subheader]
    assert "Seen movie" not in [item.value for item in app.subheader]
    assert len(app.metric) == 3
    app.selectbox[0].set_value("Drama")
    app.slider[1].set_value(5.0)
    app.button[0].click().run()
    assert not app.exception
    assert any("No unrated movies" in item.value for item in app.info)
    st.cache_resource.clear()


@pytest.mark.parametrize("invalid_pickle", [False, True])
def test_streamlit_model_errors_are_actionable(
    data_dir, tmp_path, monkeypatch, invalid_pickle
):
    path = tmp_path / "model.pkl"
    if invalid_pickle:
        path.write_bytes(b"not a pickle")
    monkeypatch.setattr(config, "DEFAULT_DATA_DIR", data_dir)
    monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", path)
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "app.py"))
    app.run()
    assert not app.exception
    assert "Could not load Popcorn Picks" in app.error[0].value
    st.cache_resource.clear()
