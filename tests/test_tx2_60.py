# test_tx2_60.py
# Including libraries
import os
import glob
import numpy as np
import pytest
from tx2_60 import TX2_60

# -----------------------------------------------------------------------------------#
# Expected configuration (from the Staubli TX2-60 CAD/datasheet)
EXPECTED_QLIM_DEG = [
    (-180.0, 180.0),
    (-127.5, 127.5),
    (-152.5, 152.5),
    (-270.0, 270.0),
    (-121.0, 132.5),
    (-270.0, 270.0),
]

EXPECTED_FLANGE_AT_HOME = np.array([0.0, 0.02, 1.045])
MESH_EXTENSIONS = (".stl", ".STL", ".dae", ".DAE", ".ply", ".PLY")


@pytest.fixture
def robot():
    return TX2_60()


def _mesh_dir():
    import tx2_60
    return os.path.join(os.path.abspath(os.path.dirname(tx2_60.__file__)), "TX2_60")


def _find_mesh(basename):
    """Find a mesh file matching basename with any extension in MESH_EXTENSIONS."""
    for ext in MESH_EXTENSIONS:
        path = os.path.join(_mesh_dir(), basename + ext)
        if os.path.isfile(path):
            return path
    matches = glob.glob(os.path.join(_mesh_dir(), basename + ".*"))
    return matches[0] if matches else None


# ------------------------------- Mesh file integrity --------------------------------#
def test_mesh_dir_exists():
    assert os.path.isdir(_mesh_dir()), f"Mesh folder not found: {_mesh_dir()}"


@pytest.mark.parametrize("link_name", [f"TX2_60Link{i}" for i in range(7)])
def test_mesh_file_exists_for_each_link(link_name):
    path = _find_mesh(link_name)
    assert path is not None, f"Missing mesh file for '{link_name}' in {_mesh_dir()}"


@pytest.mark.parametrize("link_name", [f"TX2_60Link{i}" for i in range(7)])
def test_mesh_file_not_empty(link_name):
    path = _find_mesh(link_name)
    if path is not None:
        assert os.path.getsize(path) > 0, f"Mesh file is empty: {path}"


def test_all_seven_links_have_distinct_mesh_files():
    paths = [_find_mesh(f"TX2_60Link{i}") for i in range(7)]
    paths = [p for p in paths if p is not None]
    assert len(set(paths)) == len(paths), "Two or more links are sharing the same mesh file"


# -------------------------------- Robot structure ------------------------------------#
def test_robot_constructs_without_error():
    TX2_60()


def test_robot_has_six_joints(robot):
    assert robot.n == 6, f"Expected 6 joints, got {robot.n}"


def test_robot_name(robot):
    assert robot.name == "StaubliTX2_60"


# --------------------------------- Joint limits ---------------------------------------#
@pytest.mark.parametrize("i,expected", list(enumerate(EXPECTED_QLIM_DEG)))
def test_joint_limits_match_datasheet(robot, i, expected):
    lo_deg = np.degrees(robot.qlim[0, i])
    hi_deg = np.degrees(robot.qlim[1, i])
    exp_lo, exp_hi = expected
    assert lo_deg == pytest.approx(exp_lo, abs=0.1), f"Joint {i+1} lower limit mismatch"
    assert hi_deg == pytest.approx(exp_hi, abs=0.1), f"Joint {i+1} upper limit mismatch"


# ------------------------------- DH parameter sanity ----------------------------------#
def test_dh_values_are_in_metre_range(robot):
    """Catches a forgotten mm -> m conversion (mm values fall outside a sane range)."""
    for link in robot.links:
        for val, name in [(link.d, "d"), (link.a, "a")]:
            if abs(val) > 1e-9:
                assert 0.005 < abs(val) < 1.0, (
                    f"Suspicious {name}={val} on a link - looks like an "
                    f"unconverted millimetre value, expected metres."
                )


def test_total_reach_matches_datasheet(robot):
    total_d = sum(abs(link.d) for link in robot.links)
    total_a = sum(abs(link.a) for link in robot.links)
    approx_reach = total_d + total_a
    assert 0.4 < approx_reach < 1.3, (
        f"Approximate reach {approx_reach:.3f}m is far from the "
        f"datasheet's ~0.67m - check for a units or DH mistake."
    )


def test_shoulder_offset_present(robot):
    d3 = robot.links[2].d
    assert d3 == pytest.approx(0.020, abs=1e-6), (
        f"Expected the shoulder offset d3=0.020 on link 3, got d3={d3}"
    )


# ----------------------------- Forward kinematics sanity -------------------------------#
def test_fk_at_home_is_finite(robot):
    T = robot.fkine(robot.q)
    assert np.all(np.isfinite(T.A)), "FK at home pose contains NaN/inf"


def test_fk_translation_within_plausible_bounds(robot):
    T = robot.fkine(robot.q)
    translation = T.t
    assert np.all(np.abs(translation) < 1.5), (
        f"End-effector at home pose is at {translation}, "
        f"which is implausibly far from the base - check DH units."
    )


def test_fk_matches_expected_home_position(robot):
    T = robot.fkine(robot.q)
    np.testing.assert_allclose(
        T.t, EXPECTED_FLANGE_AT_HOME, atol=0.01,
        err_msg=(
            f"Flange position at q=0 is {T.t}, differs from the expected "
            f"{EXPECTED_FLANGE_AT_HOME} - the DH parameters may not match "
            f"the mesh/CAD source."
        ),
    )


def test_fk_changes_with_joint_motion(robot):
    T_home = robot.fkine(robot.q)
    q_moved = np.array([30, -20, 40, 0, 30, 0]) * np.pi / 180.0
    T_moved = robot.fkine(q_moved)
    assert not np.allclose(T_home.A, T_moved.A), (
        "FK did not change after moving joints - the DH chain may be broken."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))