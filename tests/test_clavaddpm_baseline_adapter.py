from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from baselines.clavaddpm.runner import ClavaDDPMMovieLensRunner


def test_clava_adapter_uses_train_only_and_preserves_explicit_repeated_events(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    interactions = pd.DataFrame(
        [
            [1, 10, 20, "2020-01-01T00:00:00Z", 4.0, "train"],
            [2, 10, 20, "2020-01-02T00:00:00Z", 5.0, "train"],
            [3, 11, 21, "2020-02-01T00:00:00Z", 3.0, "validation"],
        ],
        columns=["event_id", "user_id", "movie_id", "event_time", "rating", "split"],
    )
    interactions.to_csv(data / "interactions.csv", index=False)
    pd.DataFrame({"user_id": [10, 11]}).to_csv(data / "users.csv", index=False)
    pd.DataFrame({"movie_id": [20, 21]}).to_csv(data / "movies.csv", index=False)
    config = {
        "seed": 42,
        "output_dir": str(tmp_path / "output"),
        "official": {
            "repository": "unused",
            "commit": "0" * 40,
            "checkout_dir": str(tmp_path / "checkout"),
            "python_executable": "auto",
        },
        "data": {
            "interactions": str(data / "interactions.csv"),
            "users": str(data / "users.csv"),
            "movies": str(data / "movies.csv"),
            "evaluation_config": "unused.yaml",
            "split": "train",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    runner = ClavaDDPMMovieLensRunner(path)
    manifest = runner.prepare_data()
    assert manifest["num_rating_events"] == 2
    assert manifest["repeated_pair_rows"] == 2
    events = pd.read_csv(runner.adapted_data / "rating_event.csv")
    assert events["rating_event_id"].tolist() == [1, 2]
    assert events["event_time"].dtype.kind in "fi"
    metadata = json.loads((runner.adapted_data / "dataset_meta.json").read_text())
    assert metadata["tables"]["rating_event"]["parents"] == ["user", "movie"]
    domain = json.loads((runner.adapted_data / "rating_event_domain.json").read_text())
    assert domain["event_time"]["type"] == "continuous"
    assert domain["rating"]["type"] == "discrete"

