import pandas as pd
from typing import ClassVar
from pysad.core.base_model import BaseModel
from pysad.utils.array_streamer import ArrayStreamer
import logging


def search_best_eval(model_cls, X, y, metric, search_space, args: dict = {}):
    """Evaluate `method` on `dataset`

    Args:
        method ([type]): [description]
        X ([type]): [description]
        y ([type]): [description]
        metric ([type]): [description]
        search_space ([type]): [description]
    """

    nb_anomalies = len(y)  # May be?

    def objective(_args: dict):
        _args.update(args)
        model: BaseModel = model_cls(**_args)

        if "initial_window_X" in _args:
            model.fit(X[: _args["initial_window_X"]])
