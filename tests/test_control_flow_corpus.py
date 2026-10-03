"""Control-flow shapes from rust-gpu's test corpus, run and checked.

The cases follow rust-gpu's ``tests/compiletests/ui/lang/control_flow``
(MIT or Apache-2.0), one function per file there. rust-gpu only compiles
them; here each loop makes progress so that it ends, and each function
returns a value that depends on the path taken, so the kernel's results
can be compared with plain Python for many inputs on every device.
"""

import numpy as np
import pytest

import numba_vulkan as nv

CASES = {
    "if": """
def f(i):
    r = 1
    if i > 0:
        r = 2
    return r
""",
    "if_else": """
def f(i):
    if i > 0:
        r = 1
    else:
        r = 2
    return r
""",
    "if_else_if_else": """
def f(i):
    if i > 0:
        r = 1
    elif i < 0:
        r = 2
    else:
        r = 3
    return r
""",
    "if_if": """
def f(i):
    r = 0
    if i > 0:
        r += 1
        if i < 10:
            r += 10
    return r
""",
    "ifx2": """
def f(i):
    r = 0
    if i > 0:
        r += 1
    if i > 1:
        r += 10
    return r
""",
    "if_return_else_return": """
def f(i):
    if i < 10:
        return 1
    else:
        return 2
""",
    "if_return_else": """
def f(i):
    if i < 10:
        return 1
    else:
        i += 5
    return i
""",
    "if_while": """
def f(i):
    if i == 0 or i > 20:
        while i < 30:
            i += 3
    return i
""",
    "for_range": """
def f(i):
    r = 0
    for k in range(i):
        r += k
    return r
""",
    "for_range_signed": """
def f(i):
    r = 0
    for k in range(-5, i):
        r += k
    return r
""",
    "for_range_step": """
def f(i):
    r = 0
    for k in range(i, -10, -3):
        r += k
    return r
""",
    "loop": """
def f(i):
    n = 0
    while True:
        n += 1
        if n > i:
            break
    return n
""",
    "while": """
def f(i):
    while i < 10:
        i += 1
    return i
""",
    "while_break": """
def f(i):
    n = 0
    while i < 10:
        n += 1
        break
    return n
""",
    "while_continue": """
def f(i):
    n = 0
    while i < 10:
        i += 1
        if i % 2 == 0:
            continue
        n += 1
    return n * 100 + i
""",
    "while_if_break_else_break": """
def f(i):
    n = 0
    while i < 10:
        n += 1
        if i == 0:
            n += 10
            break
        else:
            n += 20
            break
    return n
""",
    "while_if_break_if_break": """
def f(i):
    n = 0
    while i < 10:
        n += 1
        if i == 0:
            break
        if i == 1:
            n += 100
            break
        i += 1
    return n
""",
    "while_if_break": """
def f(i):
    n = 0
    while i < 10:
        n += 1
        if i == 3:
            break
        i += 1
    return n
""",
    "while_if_continue_else_continue": """
def f(i):
    n = 0
    while i < 10:
        i += 1
        if i == 4:
            n += 1
            continue
        else:
            n += 2
            continue
    return n
""",
    "while_if_continue": """
def f(i):
    n = 0
    while i < 10:
        i += 1
        if i == 4:
            continue
        n += 1
    return n
""",
    "while_return": """
def f(i):
    while i < 10:
        return i * 2
    return -1
""",
    "while_while": """
def f(i):
    n = 0
    while i < 20:
        while i < 10:
            i += 1
            n += 1
        i += 2
        n += 10
    return n
""",
    "while_while_break": """
def f(i):
    n = 0
    while i < 20:
        while i < 10:
            n += 1
            break
        i += 1
    return n
""",
    "while_while_continue": """
def f(i):
    n = 0
    while i < 20:
        while i < 10:
            i += 1
            if i % 3 == 0:
                continue
            n += 1
        i += 1
    return n
""",
    "while_while_if_break": """
def f(i):
    n = 0
    while i < 20:
        while i < 10:
            i += 1
            if i > 6:
                break
            n += 1
        i += 1
        n += 100
    return n
""",
    "while_while_if_continue": """
def f(i):
    n = 0
    while i < 20:
        while i < 10:
            i += 1
            if i > 5:
                continue
            n += 1
        i += 1
    return n
""",
    "defer": """
def f(i):
    n = 0
    while i < 32:
        current = 3
        if i < current:
            n += 1
            break
        if i < current * 2:
            n += 2
            break
        i += 4
    return n * 100 + i
""",
    "closure_multi": """
def visit(r, i, k):
    if r == k:
        return -1
    return r + i

def f(i):
    r = 0
    for k in range(10):
        r = visit(r, i, k)
        if r < 0:
            return k
    return r
""",
}

INPUTS = np.arange(-8, 40, dtype=np.int32)


@pytest.mark.parametrize("name", sorted(CASES))
def test_case(name, run):
    scope = {}
    exec(CASES[name], scope)  # noqa: S102
    want = np.array([scope["f"](int(i)) for i in INPUTS], dtype=np.int32)
    for helper in [k for k in scope if not k.startswith("__")]:
        scope[helper] = nv.jit(scope[helper])
    helper = scope["f"]

    @nv.jit
    def kernel(values, out):
        k = nv.global_id(0)
        if k < values.shape[0]:
            out[k] = helper(values[k])

    out = np.zeros_like(INPUTS)
    run(kernel, INPUTS.size, INPUTS, out)
    np.testing.assert_array_equal(out, want)
