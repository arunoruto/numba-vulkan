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


def test_loop_with_two_exit_targets_gets_one_exit_block():
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
    exits = {
        s
        for n in graph.loops["head"]
        for s in graph.succs[n]
        if s not in graph.loops["head"]
    }
    assert exits == {"head.exit"}
