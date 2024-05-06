import json
from pprint import pprint
import pandas as pd
import pandas_dedupe

with open("mixotic.json", "r") as f:
    mixes = json.load(f)

alltracks = pd.json_normalize(
    mixes,
    record_path=["playlist"],
    meta=["id", "title", "artist"],
    record_prefix="track.",
    meta_prefix="mix.",
)

dedupe = pandas_dedupe.dedupe_dataframe(
    alltracks,
    ["track.release", "track.extra"],
    # update_model=True
)

dedupe = dedupe.rename(
    columns={"cluster id": "track.labelid", "confidence": "track.labelid.confidence"}
)

dedupe.to_csv("mixotic-cluster-by-label.csv")
dedupe.to_json("mixotic-cluster-by-label.json")
