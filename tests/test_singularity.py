import os
import sys
import numpy as np

#Alloww test/to import files from AT 2
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from safety import SingularitySafety
from kawasaki_rs007n_task import KawasakiPickPlace
from staubli_tx2_60_task import StaubliShapeDraw
from nachi_mz07_task import NachiShapeFill


#Kawwasaki RS007N

def test_kawasaki_singularity():
    arm = KawasakiPickPlace()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert isinstance(result, dict)
    assert isinstance(result["safe"], bool)
    assert result["level"] in ["SAFE","WARNING","STOP"]
    assert result["sigma_min"] >= 0
    assert 0 <= result["rank"] <= 6

#Staubli TX2-60

def test_staubli_singularity():
    arm = StaubliShapeDraw()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "level" in result
    assert "sigma_min" in result
    assert "condition_number" in result
    assert "manipulability" in result
    assert "rank" in result

def test_staubli_singularity_value_valid():
    arm = StaubliShapeDraw()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert result["sigma_min"] >= 0
    assert result["rank"] >= 0  
    assert result["rank"] <= 6

    assert result ["level"] in ["SAFE", "WARNING", "STOP"]

#Nachi MZ07
def test_nachi_singularity():
    arm = NachiShapeFill()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "level" in result
    assert "sigma_min" in result
    assert "condition_number" in result
    assert "manipulability" in result
    assert "rank" in result

def test_nachi_singularity_value_valid():
    arm = NachiShapeFill()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert result["sigma_min"] >= 0
    assert result["rank"] >= 0
    assert result["rank"] <= 6

    assert result ["level"] in ["SAFE", "WARNING", "STOP"]

#ARTIFICIAL SINGULARITY TESTS

def test_singularity_metrics_at_zero_configuration():

    """
    Check that the singularity monitor can analyse q = 0 without crashing.

    This does not assume q = 0 is necessarily singular for every robot.

    """
    arm = KawasakiPickPlace()
    safety = SingularitySafety()

    q_zero = np.zeros(6)
    result = safety.check(arm, q_zero)

    assert isinstance(result, dict)
    assert result["sigma_min"] >= 0
    assert result["rank"] >= 0
    assert result["rank"] <= 6 