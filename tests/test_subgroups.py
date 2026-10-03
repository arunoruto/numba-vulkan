"""Subgroup operations, checked against the subgroups the device formed.

Vulkan does not say which invocations form a subgroup, so every kernel also
reports its subgroup and lane, and the expected values are computed from the
members of each subgroup as the device formed them.
"""

import numpy as np
import pytest
from numba import types

import numba_vulkan as nv
from numba_vulkan import codegen
from numba_vulkan.errors import VulkanSupportError

N, LOCAL = 256, 64
sg = nv.subgroup


def _launch(device, kernel, *arrays):
    """Run over N invocations in workgroups of LOCAL, or skip if unsupported."""
    try:
        kernel[N // LOCAL, LOCAL, device](*arrays)
    except VulkanSupportError as exc:
        pytest.skip(str(exc))


@nv.jit
def where(out):
    i = nv.global_id(0)
    out[i, 0] = sg.id()
    out[i, 1] = sg.lane()
    out[i, 2] = sg.size()
    out[i, 3] = sg.count()


def _members(device):
    """Global indices of the invocations in each subgroup, by lane."""
    out = np.zeros((N, 4), np.int32)
    _launch(device, where, out)
    groups = {}
    for i in range(N):
        key = (i // LOCAL, out[i, 0])
        groups.setdefault(key, {})[out[i, 1]] = i
    return out, list(groups.values())


def test_identities(device):
    out, groups = _members(device)
    size = nv.list_devices()[device].subgroup_size
    assert (out[:, 2] == size).all()
    # Subgroups need not be full: Intel's driver runs this kernel 16 wide
    # while reporting a size of 32.
    assert (out[:, 3] == len(groups) // (N // LOCAL)).all()
    for lanes in groups:
        assert max(lanes) < size and min(lanes) == 0


def _reduce_kernel(name):
    scope = {"sg": sg, "nv": nv}
    exec(  # noqa: S102
        f"def kernel(x, out):\n    i = nv.global_id(0)\n    out[i] = sg.{name}(x[i])\n",
        scope,
    )
    return nv.jit(scope["kernel"])


OPS = {
    "sum": (np.sum, 0),
    "prod": (np.prod, 1),
    "min": (np.min, None),
    "max": (np.max, None),
}


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32, np.float64])
@pytest.mark.parametrize("op", sorted(OPS))
def test_reductions_and_scans(device, op, dtype):
    _, groups = _members(device)
    rng = np.random.default_rng(1)
    if op == "prod":
        values = rng.choice([1, 2], N).astype(dtype)  # no overflow
    elif dtype == np.uint32:
        values = rng.integers(0, 1 << 31, N).astype(dtype)
    else:
        values = rng.integers(-1000, 1000, N).astype(dtype)  # exact sums
    combine, _ = OPS[op]
    for kind in ("", "inclusive_", "exclusive_"):
        out = np.zeros(N, dtype)
        _launch(device, _reduce_kernel(kind + op), values, out)
        for lanes in groups:
            for lane, i in lanes.items():
                if kind == "":
                    taken = list(lanes.values())
                elif kind == "inclusive_":
                    taken = [lanes[k] for k in lanes if k <= lane]
                else:
                    taken = [lanes[k] for k in lanes if k < lane]
                if not taken:
                    continue  # the identity; see the next test
                want = combine(values[taken]).astype(dtype)
                assert out[i] == want, (kind + op, lane, out[i], want)


def test_exclusive_scans_start_with_the_identity(device):
    _, groups = _members(device)
    values = np.arange(N, dtype=np.float32) + 5
    first = [lanes[0] for lanes in groups]
    expected = {"sum": 0, "prod": 1, "min": np.inf, "max": -np.inf}
    for op, identity in expected.items():
        out = np.zeros(N, np.float32)
        _launch(device, _reduce_kernel(f"exclusive_{op}"), values, out)
        assert (out[first] == identity).all(), op


@nv.jit
def exchanges(x, out):
    i = nv.global_id(0)
    lane = sg.lane()
    out[i, 0] = sg.broadcast(x[i], 0)
    out[i, 1] = sg.broadcast_first(x[i])
    out[i, 2] = sg.shuffle(x[i], (lane + 1) % sg.size())
    out[i, 3] = sg.shuffle_xor(x[i], 1)
    out[i, 4] = sg.shuffle_up(x[i], 1)
    out[i, 5] = sg.shuffle_down(x[i], 1)


def test_exchanges(device):
    _, groups = _members(device)
    x = np.arange(N, dtype=np.float32) * 3 + 1
    out = np.zeros((N, 6), np.float32)
    _launch(device, exchanges, x, out)
    size = nv.list_devices()[device].subgroup_size
    for lanes in groups:
        # Only lanes that exist can be read from; subgroups may be partial.
        for lane, i in lanes.items():
            assert out[i, 0] == x[lanes[0]]
            assert out[i, 1] == x[lanes[0]]
            for column, source in ((2, (lane + 1) % size), (3, lane ^ 1)):
                if source in lanes:
                    assert out[i, column] == x[lanes[source]]
            if lane - 1 in lanes:
                assert out[i, 4] == x[lanes[lane - 1]]
            if lane + 1 in lanes:
                assert out[i, 5] == x[lanes[lane + 1]]


@nv.jit
def votes(x, out):
    i = nv.global_id(0)
    p = x[i] > 0
    out[i, 0] = sg.any(p)
    out[i, 1] = sg.all(p)
    out[i, 2] = sg.elect()
    out[i, 3] = sg.ballot_count(p)
    words = sg.ballot(p)
    out[i, 4] = words[0]
    out[i, 5] = words[1]


def test_votes_and_ballots(device):
    _, groups = _members(device)
    x = np.where(np.arange(N) % 7 == 3, 1, -1).astype(np.int32)
    x[: LOCAL // 2] = 1  # a subgroup where all hold, on most devices
    out = np.zeros((N, 6), np.uint32)
    _launch(device, votes, x, out)
    for lanes in groups:
        held = {lane: x[i] > 0 for lane, i in lanes.items()}
        mask = sum(1 << int(lane) for lane, h in held.items() if h)
        for lane, i in lanes.items():
            assert out[i, 0] == any(held.values())
            assert out[i, 1] == all(held.values())
            assert out[i, 2] == (lane == 0)
            assert out[i, 3] == sum(held.values())
            assert out[i, 4] == mask & 0xFFFFFFFF
            assert out[i, 5] == (mask >> 32) & 0xFFFFFFFF


def test_under_a_condition_only_active_invocations_take_part(device):
    @nv.jit
    def odd_sum(x, out):
        i = nv.global_id(0)
        if sg.lane() % 2 == 1:
            out[i] = sg.sum(x[i])

    _, groups = _members(device)
    x = np.arange(N, dtype=np.int32)
    out = np.full(N, -1, np.int32)
    _launch(device, odd_sum, x, out)
    for lanes in groups:
        odd = [i for lane, i in lanes.items() if lane % 2]
        for i in odd:
            assert out[i] == x[odd].sum()


def test_operations_are_never_copied_by_the_structurizer(device):
    @nv.jit
    def early(x, out):
        i = nv.global_id(0)
        v = x[i]
        if v > 100 or v < 3:
            v = v * 2
        out[i] = sg.sum(v)

    _, groups = _members(device)
    x = np.arange(N, dtype=np.int32)
    out = np.zeros(N, np.int32)
    _launch(device, early, x, out)
    doubled = np.where((x > 100) | (x < 3), x * 2, x)
    for lanes in groups:
        members = list(lanes.values())
        assert (out[members] == doubled[members].sum()).all()


def test_modules_declare_what_they_use(validate):
    kernel = exchanges.compile((types.float32[::1], types.float32[:, ::1]), 1)
    validate(kernel)
    assert {"subgroup_shuffle", "subgroup_shuffle_relative", "subgroup_ballot"} <= (
        kernel.capabilities
    )
    _, instructions = codegen._instructions(kernel.spirv)
    assert not any(i[0] & 0xFFFF == 17 and i[1] == 5 for i in instructions)  # Linkage


def test_unsupported_types_are_rejected():
    @nv.jit
    def small(x, out):
        out[0] = sg.sum(x[0])

    with pytest.raises(Exception, match="32- or 64-bit integers or floats"):
        small.compile((types.int8[::1], types.int8[::1]), 1)
