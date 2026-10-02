"""Unit tests of the control-flow restructuring on hand-written LLVM IR."""

import llvmlite.binding as llvm

from numba_vulkan.structurize import _Graph, _parse, structurize

HEADER = "define i32 @f(i1 %a, i1 %b, i1 %c) {\n"


def blocks_of(text):
    body = text.split("{\n", 1)[1].rsplit("}", 1)[0]
    return _parse(body)


def assert_structured(text):
    """Every join must be the merge block of the selection that dominates it,
    and no selection may share its merge block with code outside of it."""
    llvm.parse_assembly(text).verify()
    blocks = blocks_of(text)
    graph = _Graph(blocks)
    for name in graph.order:
        if len(graph.preds[name]) > 1 and name not in graph.headers:
            assert graph.ipdom[graph.idom[name]] == name, (
                f"{name} is an unstructured join"
            )
    for header in graph.order:
        merge = graph.ipdom.get(header)
        if len(graph.succs[header]) > 1 and merge in blocks:
            if graph.loop_of[header] == graph.loop_of[merge] != merge:
                outside = [
                    p for p in graph.preds[merge] if not graph.dominates(header, p)
                ]
                assert not outside, f"{header} shares its merge block {merge}"


EARLY_RETURNS = (
    HEADER
    + """\
entry:
  br i1 %a, label %done, label %second
second:
  br i1 %b, label %done, label %third
third:
  br i1 %c, label %done, label %last
last:
  br label %done
done:
  %r = phi i32 [ 1, %entry ], [ 2, %second ], [ 3, %third ], [ 4, %last ]
  ret i32 %r
}
"""
)

SHORT_CIRCUIT = (
    HEADER
    + """\
entry:
  br i1 %a, label %then, label %test
test:
  br i1 %b, label %then, label %else
then:
  %x = add i32 1, 2
  br label %done
else:
  br label %done
done:
  %r = phi i32 [ %x, %then ], [ 7, %else ]
  ret i32 %r
}
"""
)

STRUCTURED = (
    HEADER
    + """\
entry:
  br i1 %a, label %then, label %else
then:
  br label %done
else:
  br label %done
done:
  %r = phi i32 [ 1, %then ], [ 2, %else ]
  ret i32 %r
}
"""
)


def test_shared_merge_block_is_split():
    out = structurize(EARLY_RETURNS)
    assert_structured(out)
    # one private merge block each for the two inner selections
    assert out.count(".merge:") == 2


def test_unstructured_join_is_duplicated():
    out = structurize(SHORT_CIRCUIT)
    assert_structured(out)
    assert "then.dup1:" in out
    # the value computed in the duplicated block reaches the phi from both copies
    assert "%x.dup1" in out


def test_structured_code_is_left_untouched():
    assert structurize(STRUCTURED) == STRUCTURED


def test_loop_with_two_exit_targets_leaves_through_its_latch():
    text = (
        HEADER
        + """\
entry:
  %slot = alloca i32
  br label %head
head:
  br i1 %a, label %body, label %after
body:
  br i1 %b, label %early, label %head
early:
  store i32 5, ptr %slot
  br label %out
after:
  store i32 6, ptr %slot
  br label %out
out:
  %r = load i32, ptr %slot
  ret i32 %r
}
"""
    )
    out = structurize(text)
    assert_structured(out)
    graph = _Graph(blocks_of(out))
    loop = graph.loops["head.head"]
    exits = {(n, s) for n in loop for s in graph.succs[n] if s not in loop}
    assert exits == {("head.latch", "head.exit")}


def assert_same_results(text, out):
    """Run the function before and after restructuring for all inputs."""
    from test_legalize import compile_ir

    types = ("i32", ["i1", "i1", "i1"])
    before, after = compile_ir(text, *types), compile_ir(out, *types)
    for bits in range(8):
        args = [bool(bits & 1), bool(bits & 2), bool(bits & 4)]
        assert before(*args) == after(*args), args


# The join is bypassed by the early exit in `inner`.
BYPASSED_JOIN = (
    HEADER
    + """\
entry:
  %slot = alloca i32
  store i32 0, ptr %slot
  br i1 %a, label %inner, label %join
inner:
  br i1 %b, label %early, label %work
early:
  store i32 1, ptr %slot
  br label %out
work:
  store i32 2, ptr %slot
  br label %join
join:
  %v = load i32, ptr %slot
  %w = add i32 %v, 10
  store i32 %w, ptr %slot
  br i1 %c, label %more, label %out
more:
  %x = mul i32 %w, 3
  store i32 %x, ptr %slot
  br label %out
out:
  %r = load i32, ptr %slot
  ret i32 %r
}
"""
)


def test_bypassed_join_is_copied_when_small():
    out = structurize(BYPASSED_JOIN)
    assert_structured(out)
    assert "join.dup1:" in out and "guard" not in out
    assert_same_results(BYPASSED_JOIN, out)


def test_bypassed_join_is_guarded_when_large(monkeypatch):
    monkeypatch.setattr("numba_vulkan.structurize._COPY_LIMIT", 0)
    out = structurize(BYPASSED_JOIN)
    assert_structured(out)
    assert "join.guard:" in out and ".dup" not in out
    assert_same_results(BYPASSED_JOIN, out)


def test_loop_left_from_nested_positions_keeps_its_meaning():
    # Counts to four; leaves early with a return (`early`) or a break (`brk`).
    text = (
        HEADER
        + """\
entry:
  %i = alloca i32
  %slot = alloca i32
  store i32 0, ptr %i
  store i32 0, ptr %slot
  br label %head
head:
  %iv = load i32, ptr %i
  %go = icmp slt i32 %iv, 4
  br i1 %go, label %body, label %after
body:
  %n = add i32 %iv, 1
  store i32 %n, ptr %i
  br i1 %a, label %check, label %latch
check:
  %two = icmp eq i32 %n, 2
  br i1 %two, label %early, label %cont
cont:
  %three = icmp eq i32 %n, 3
  %stop = and i1 %three, %b
  br i1 %stop, label %brk, label %latch
latch:
  %s = load i32, ptr %slot
  %t = add i32 %s, %n
  store i32 %t, ptr %slot
  br label %head
early:
  br i1 %c, label %ret, label %brk
ret:
  ret i32 -1
brk:
  %u = load i32, ptr %slot
  %y = mul i32 %u, 100
  store i32 %y, ptr %slot
  br label %after
after:
  %r = load i32, ptr %slot
  ret i32 %r
}
"""
    )
    out = structurize(text)
    assert_structured(out)
    assert_same_results(text, out)
