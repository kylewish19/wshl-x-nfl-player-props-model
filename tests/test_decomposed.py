import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from wshlx_nfl_props.decomposed import DecomposedYardageModel


def test_decomposed_predicts_point_and_probability():
    x=pd.DataFrame({"f":[1.0,2.0,3.0]})
    vp=DummyRegressor(strategy="constant",constant=20.0).fit(x,[1,1,1])
    ep=DummyRegressor(strategy="constant",constant=5.0).fit(x,[1,1,1])
    m=DecomposedYardageModel(
        prop="rb_rush_yds",kind="regression",estimator_name="decomposed_test",
        feature_set="baseline",volume_pipeline=vp,efficiency_pipeline=ep,
        numeric_features=["f"],categorical_features=[],volume_target="rushing_attempts",
        yardage_target="rushing_yards",efficiency_clip=(-2.0,12.0),
        residuals=[-10,0,10]*10,validation_metrics={}
    )
    point=m.predict_point(x.iloc[[0]])[0]
    prob=m.probability_over(x.iloc[[0]],95.0)[0]
    assert point==100.0
    assert 0.0 <= prob <= 1.0
