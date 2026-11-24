"""
conftest.py for Issue #50 yield forecast tests.
"""
import os
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def synthetic_sample_data():
    """Generates a small synthetic fixture mimicking the dataset for unit testing."""
    np.random.seed(42)
    records = []
    districts = [502, 503, 505]
    crops = ["rice", "groundnut", "maize"]
    years = [2017, 2018, 2019]
    
    for yr in years:
        for d in districts:
            for c in crops:
                records.append({
                    "district_code": d,
                    "district_name": f"District_{d}",
                    "crop": c,
                    "crop_name": c.capitalize(),
                    "year": yr,
                    "yield_kg_per_ha": 2000 + np.random.normal(0, 300),
                    "yield_anomaly_kg_per_ha": np.random.normal(0, 150),
                    "proceed_to_modelling": True,
                    "is_eligible_crop": True,
                    "has_valid_yield_anomaly": True,
                    "usable_for_forward_chaining_eval": True,
                    "climate_zone": f"Climate Zone {(d % 3) + 1}",
                    "cluster_id": d % 3,
                    "wanom__annual_cal__rain_sum": np.random.normal(0, 50),
                    "wanom__kharif__rain_sum": np.random.normal(0, 40),
                    "wanom__annual_crop__hot_days": np.random.normal(0, 5),
                    "wanom__ne__rainy_days": np.random.normal(0, 3),
                })
    return pd.DataFrame(records)
