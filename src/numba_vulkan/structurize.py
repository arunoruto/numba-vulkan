"""Restructure control flow before SPIR-V code generation.

SPIR-V requires structured control flow: each conditional branch declares a
merge block, selections must nest properly, and a block inside a selection
may only leave it through that selection's merge block. Ordinary Python
breaks this shape in two ways.

Early exits. In ::

    if a: return x
    if b: return y
    return z

all three returns jump to the same continuation from different nesting
depths, so several selections share one merge block.

Short-circuit conditions. In ::

    if a or b: X
    else: Y

the block X is entered from the test of ``a`` and from the test of ``b``,
which is a join that is not the merge block of any selection.

LLVM's SPIR-V structurizer is meant to repair such graphs, but in LLVM 22
it emits invalid modules for many of them. `structurize` therefore brings
the graph into properly nested form itself, in three steps:

0. `_unify_loop_exits` gives every loop a single exit block;
1. `_duplicate_joins` copies every block that is entered from several
   places without being a merge block, once per entry, so that only proper
   merge blocks have more than one predecessor;
2. `_split_merges` gives each selection that shares its merge block with
   an enclosing one a fresh merge block that forwards to the shared one.

Loops. A ``return`` inside a loop leaves it towards a block that is not
the loop's merge block. `_unify_loop_exits` routes all exits of a loop
through one exit block and dispatches from there, which runs first.

The transformation works on textual LLVM IR, because it has to run after
all LLVM passes and llvmlite offers no API for editing a module.
"""

import re

_NAME = r'%(?:"[^"]+"|[-a-zA-Z$._0-9]+)'
_LABEL_LINE = re.compile(r'^("[^"]+"|[-a-zA-Z$._0-9]+):')
_LABEL_REF = re.compile(rf"label ({_NAME})")
_PHI = re.compile(rf"^(\s*)({_NAME}) = phi (.+?) (\[ .*)$")
_PHI_ENTRY = re.compile(rf"\[ (.+?), ({_NAME}) \]")
_TOKEN = re.compile(_NAME)
_CONDITIONAL = re.compile(rf"\s*br i1 (\S+), label ({_NAME}), label ({_NAME})")
# Names that need no quotes: identifiers, or the numbers of unnamed values.
_PLAIN = re.compile(r"[-a-zA-Z$._][-a-zA-Z$._0-9]*|[0-9]+")
_DEF = re.compile(rf"^\s*({_NAME}) = ")
# Duplicating joins can grow a function exponentially. Beyond this size the
# graph is handed to LLVM as it is, which then usually reports an error.
_MAX_BLOCKS = 600
_DEFINE = re.compile(r"^define [^\n]*\{\n(.*?)^\}", re.MULTILINE | re.DOTALL)


class _Block:
    """One basic block: its phi lines, other instructions and terminator."""

    def __init__(self, name):
        self.name = name
        self.phis = []
        self.body = []
        self.term = []

    @property
    def successors(self):
        """Names of the blocks the terminator branches to, without duplicates.

        Returns
        -------
        list of str
        """
        seen = []
        for ref in _LABEL_REF.findall(" ".join(self.term)):
            if ref[1:].strip('"') not in seen:
                seen.append(ref[1:].strip('"'))
        return seen

    def retarget(self, old, new):
        """Make the terminator branch to `new` wherever it branched to `old`."""
        pattern = re.compile(rf"label {re.escape(_ref(old))}(?![-a-zA-Z$._0-9])")
        self.term = [
            pattern.sub(lambda m: f"label {_ref(new)}", line) for line in self.term
        ]

    def definitions(self):
        """Names of the SSA values this block defines."""
        return [
            m.group(1)[1:].strip('"')
            for m in map(_DEF.match, self.phis + self.body)
            if m
        ]

    def renamed(self, name, mapping):
        """Copy of this block with the names in `mapping` replaced.

        Parameters
        ----------
        name : str
            Name of the copy.
        mapping : dict of str to str
            New names for blocks and SSA values, without the ``%`` sigil.

        Returns
        -------
        _Block
        """

        def swap(match):
            """Replace one ``%name`` token if it has a new name."""
            raw = match.group(0)[1:].strip('"')
            return _ref(mapping[raw]) if raw in mapping else match.group(0)

        copy = _Block(name)
        copy.phis = [_TOKEN.sub(swap, line) for line in self.phis]
        copy.body = [_TOKEN.sub(swap, line) for line in self.body]
        copy.term = [_TOKEN.sub(swap, line) for line in self.term]
        return copy

    def lines(self):
        """The block as lines of LLVM IR, starting with its label.

        Returns
        -------
        list of str
        """
        label = self.name if _PLAIN.fullmatch(self.name) else f'"{self.name}"'
        return [f"{label}:"] + self.phis + self.body + self.term


def _ref(name):
    """Spell a block name as an operand."""
    return f"%{name}" if _PLAIN.fullmatch(name) else f'%"{name}"'


def _parse(body):
    """Split a function body into blocks.

    Returns
    -------
    dict of str to _Block or None
        Blocks by name in their original order, or ``None`` when the body
        has an unlabelled block and is therefore left alone.
    """
    blocks, current = {}, None
    for line in body.splitlines():
        if not line.strip():
            continue
        match = _LABEL_LINE.match(line)
        if match:
            current = _Block(match.group(1).strip('"'))
            blocks[current.name] = current
        elif current is None:
            return None
        elif current.term or re.match(r"\s*(br|ret|switch|unreachable)\b", line):
            current.term.append(line)
        elif _PHI.match(line):
            current.phis.append(line)
        else:
            current.body.append(line)
    return blocks


def _dominators(order, preds, entry):
    """Immediate dominators by the iterative data-flow algorithm.

    Parameters
    ----------
    order : list of str
        Nodes in reverse post-order from `entry`.
    preds : dict of str to list of str
        Predecessors of every node.
    entry : str
        The root.

    Returns
    -------
    dict of str to str
        Immediate dominator of each reachable node; the root maps to itself.
    """
    index = {n: i for i, n in enumerate(order)}
    idom = {entry: entry}

    def intersect(a, b):
        """Nearest common ancestor of two nodes in the dominator tree."""
        while a != b:
            while index[a] > index[b]:
                a = idom[a]
            while index[b] > index[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for node in order[1:]:
            done = [p for p in preds[node] if p in idom]
            if not done:
                continue
            new = done[0]
            for p in done[1:]:
                new = intersect(p, new)
            if idom.get(node) != new:
                idom[node] = new
                changed = True
    return idom


def _reverse_post_order(succs, entry):
    """Order the nodes reachable from the entry in reverse post-order.

    Parameters
    ----------
    succs : dict of str to list of str
        Successors of every node.
    entry : str
        The node to start from.

    Returns
    -------
    list of str
    """
    seen, out = set(), []

    def visit(node):
        """Depth-first traversal without recursion."""
        stack = [(node, iter(succs[node]))]
        seen.add(node)
        while stack:
            current, it = stack[-1]
            for nxt in it:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, iter(succs[nxt])))
                    break
            else:
                out.append(current)
                stack.pop()

    visit(entry)
    return out[::-1]


def _dominates(idom, a, b):
    """Whether `a` dominates `b` in the tree given by `idom`."""
    while True:
        if a == b:
            return True
        if b not in idom or idom[b] == b:
            return False
        b = idom[b]


def _innermost_loops(names, succs, preds, idom):
    """Find the innermost natural loop of every block.

    Parameters
    ----------
    names : list of str
        All blocks.
    succs, preds : dict of str to list of str
        The control-flow graph.
    idom : dict of str to str
        Immediate dominators.

    Returns
    -------
    loop_of : dict of str to str or None
        Header of the innermost loop containing each block, or ``None``.
    latches : set of str
        Blocks with a back edge to a loop header.
    bodies : dict of str to set of str
        The blocks of each loop, by loop header.
    """
    bodies, latches = {}, set()
    for node in names:
        for succ in succs[node]:
            if node in idom and _dominates(idom, succ, node):  # back edge
                latches.add(node)
                body = bodies.setdefault(succ, {succ})
                work = [node]
                while work:
                    current = work.pop()
                    if current not in body:
                        body.add(current)
                        work.extend(preds[current])
    loop_of = {}
    for node in names:
        containing = [h for h, body in bodies.items() if node in body]
        loop_of[node] = (
            min(containing, key=lambda h: len(bodies[h])) if containing else None
        )
    return loop_of, latches, bodies


class _Graph:
    """Control-flow facts about a function, recomputed after each change."""

    def __init__(self, blocks):
        self.names = list(blocks)
        self.entry = self.names[0]
        self.succs = {
            n: [s for s in blocks[n].successors if s in blocks] for n in self.names
        }
        self.preds = {n: [] for n in self.names}
        for n in self.names:
            for succ in self.succs[n]:
                self.preds[succ].append(n)
        self.order = _reverse_post_order(self.succs, self.entry)
        self.idom = _dominators(self.order, self.preds, self.entry)

        # Post-dominators: dominators of the reversed graph from a virtual exit.
        self.exit = "<exit>"
        rsuccs = {n: list(self.preds[n]) for n in self.names}
        rsuccs[self.exit] = [n for n in self.names if not self.succs[n]]
        rpreds = {n: list(self.succs[n]) for n in self.names}
        for n in rsuccs[self.exit]:
            rpreds[n].append(self.exit)
        rpreds[self.exit] = []
        self.ipdom = _dominators(
            _reverse_post_order(rsuccs, self.exit), rpreds, self.exit
        )

        self.loop_of, self.latches, self.loops = _innermost_loops(
            self.names, self.succs, self.preds, self.idom
        )
        self.headers = {h for h in self.loop_of.values() if h is not None}
        self.depth = {}
        for n in self.order:
            self.depth[n] = 0 if n == self.entry else self.depth[self.idom[n]] + 1

    def dominates(self, a, b):
        """Whether block `a` dominates block `b`.

        Returns
        -------
        bool
        """
        return _dominates(self.idom, a, b)


def _unify_loop_exits(blocks):
    """Give every loop a single exit block that only the loop branches to.

    A ``return`` inside a loop, or a ``break`` out of nested selections,
    leaves the loop towards a block that is not the loop's merge block. Each
    such loop gets one exit block; the edges leaving the loop record which
    target they wanted in a stack slot, and a chain of tests after the exit
    block dispatches to it. Inner loops are handled first, so an exit that
    crosses several loops is passed outwards one level at a time.

    The function requires phi-free IR (LLVM's ``reg2mem``); loops whose exit
    targets still have phi nodes are left alone.

    Returns
    -------
    bool
        Whether anything changed.
    """
    changed, done, counter = False, set(), 0
    while len(blocks) < _MAX_BLOCKS:
        graph = _Graph(blocks)
        for header in sorted(graph.loops, key=lambda h: len(graph.loops[h])):
            if header in done:
                continue
            done.add(header)
            body = graph.loops[header]
            exits = [
                (b, t)
                for b in graph.order
                if b in body
                for t in graph.succs[b]
                if t not in body
            ]
            targets = list(dict.fromkeys(t for _, t in exits))
            if not exits or any(blocks[t].phis for t in targets):
                continue
            # A block leaving towards two targets must end in a two-way branch.
            if any(
                len({t for b2, t in exits if b2 == b}) > 1
                and not _CONDITIONAL.match(blocks[b].term[0])
                for b, _ in exits
            ):
                continue
            if len(targets) == 1 and all(p in body for p in graph.preds[targets[0]]):
                continue
            break
        else:
            return changed

        counter += 1
        slot = f"%nv.exit{counter}"
        merge = f"{header}.exit"
        while merge in blocks:
            merge += "_"
        many = len(targets) > 1
        if many:
            blocks[graph.entry].body.insert(0, f"  {slot} = alloca i32")
        # The exiting blocks branch to the exit block directly, as a break
        # must. Each records its target just before leaving; a block that
        # stays in the loop on one arm stores needlessly, which is harmless
        # because the slot is only read after the loop.
        for block in dict.fromkeys(b for b, _ in exits):
            leaving = [t for b, t in exits if b == block]
            if many:
                wanted = [targets.index(t) for t in leaving]
                cond = _CONDITIONAL.match(blocks[block].term[0])
                if len(set(wanted)) == 1:
                    value = str(wanted[0])
                else:
                    first, second = (
                        cond.group(2)[1:].strip('"'),
                        cond.group(3)[1:].strip('"'),
                    )
                    value = f"{slot}.s{len(blocks)}"
                    blocks[block].body.append(
                        f"  {value} = select i1 {cond.group(1)}, "
                        f"i32 {targets.index(first)}, i32 {targets.index(second)}"
                    )
                blocks[block].body.append(f"  store i32 {value}, ptr {slot}")
            for target in leaving:
                blocks[block].retarget(target, merge)
        current = _Block(merge)
        blocks[merge] = current
        for k, target in enumerate(targets[:-1]):
            nxt = _Block(f"{merge}.test{k + 1}")
            # Every test reloads the slot, so no value crosses a block.
            current.body = [
                f"  {slot}.v{k} = load i32, ptr {slot}",
                f"  {slot}.c{k} = icmp eq i32 {slot}.v{k}, {k}",
            ]
            current.term = [
                f"  br i1 {slot}.c{k}, label {_ref(target)}, label {_ref(nxt.name)}"
            ]
            if k + 1 < len(targets) - 1:
                blocks[nxt.name] = nxt
                current = nxt
            else:
                # the last test falls through to the last target directly
                current.term = [
                    f"  br i1 {slot}.c{k}, label {_ref(target)}, label {_ref(targets[-1])}"
                ]
        if not current.term:
            current.term = [f"  br label {_ref(targets[-1])}"]
        changed = True
    return changed


def _unstructured_join(graph):
    """Find a block entered from several places that is no merge block.

    A join is a proper merge block when it post-dominates its immediate
    dominator: then every path out of that dominator's selection passes
    through it.

    Returns
    -------
    tuple or None
        ``(join, region)`` for the innermost such block, where ``region``
        is the set of blocks it dominates, or ``None`` if there is none (or
        none that can be duplicated safely).
    """
    for join in sorted(graph.order, key=lambda n: -graph.depth[n]):
        if len(graph.preds[join]) < 2 or join in graph.headers or join == graph.entry:
            continue
        dominator = graph.idom[join]
        if graph.ipdom.get(dominator) == join:
            continue
        if graph.loop_of[join] != graph.loop_of[dominator]:
            continue
        region = {n for n in graph.order if graph.dominates(join, n)}
        # Copying a latch would give its loop a second continue block.
        if any(
            s in graph.headers and s not in region
            for n in region
            for s in graph.succs[n]
        ):
            continue
        return join, region
    return None


def _duplicate_joins(blocks):
    """Copy unstructured joins so that each copy has a single entry.

    Returns
    -------
    bool
        Whether anything changed.
    """
    changed, copies = False, 0
    while len(blocks) < _MAX_BLOCKS:
        graph = _Graph(blocks)
        found = _unstructured_join(graph)
        if found is None:
            break
        join, region = found
        defined = [d for n in region for d in blocks[n].definitions()]
        frontier = {s for n in region for s in graph.succs[n] if s not in region}
        # The first predecessor keeps the original; every other gets a copy.
        for pred in graph.preds[join][1:]:
            copies += 1
            mapping = {n: f"{n}.dup{copies}" for n in region}
            mapping.update({d: f"{d}.dup{copies}" for d in defined})
            for n in region:
                blocks[mapping[n]] = blocks[n].renamed(mapping[n], mapping)
            twin = blocks[mapping[join]]
            twin.phis = [_keep_entries(line, lambda b: b == pred) for line in twin.phis]
            blocks[pred].retarget(join, mapping[join])
            blocks[join].phis = [
                _keep_entries(line, lambda b: b != pred) for line in blocks[join].phis
            ]
            # Blocks after the region now also receive values from the copy.
            for name in frontier:
                blocks[name].phis = [
                    _add_copies(line, region, mapping) for line in blocks[name].phis
                ]
        changed = True
    return changed


def _keep_entries(line, keep):
    """Drop the phi entries whose predecessor does not satisfy `keep`."""
    indent, res, ty, rest = _PHI.match(line).groups()
    entries = [(v, b) for v, b in _PHI_ENTRY.findall(rest) if keep(b[1:].strip('"'))]
    return f"{indent}{res} = phi {ty} " + ", ".join(f"[ {v}, {b} ]" for v, b in entries)


def _add_copies(line, region, mapping):
    """Add phi entries for the copies of the predecessors in `region`."""
    indent, res, ty, rest = _PHI.match(line).groups()
    entries = _PHI_ENTRY.findall(rest)
    extra = []
    for value, block in entries:
        raw = block[1:].strip('"')
        if raw in region:
            name = value[1:].strip('"') if value.startswith("%") else None
            extra.append(
                (_ref(mapping[name]) if name in mapping else value, _ref(mapping[raw]))
            )
    return f"{indent}{res} = phi {ty} " + ", ".join(
        f"[ {v}, {b} ]" for v, b in entries + extra
    )


def _split_merges(blocks):
    """Insert private merge blocks; returns whether anything changed."""
    graph = _Graph(blocks)
    succs, preds, idom, ipdom = graph.succs, graph.preds, graph.idom, graph.ipdom
    loop_of, latches, depth = graph.loop_of, graph.latches, graph.depth
    order, exit_ = graph.order, graph.exit

    changed = False
    # Innermost selections first, so that outer ones see the new merge blocks.
    for header in sorted(
        (n for n in order if len(succs[n]) > 1), key=lambda n: -depth[n]
    ):
        merge = ipdom.get(header)
        if merge is None or merge == exit_:
            continue
        # A selection whose paths leave its loop (break, return) is left to
        # LLVM; only selections that stay within one loop level are handled.
        if loop_of.get(header) != loop_of.get(merge) or loop_of.get(header) == merge:
            continue
        inside = [p for p in preds[merge] if _dominates(idom, header, p)]
        outside = [p for p in preds[merge] if p not in inside]
        # The merge block must not double as the continue target of a loop.
        if not inside or not (outside or merge in latches):
            continue

        name = f"{header}.merge"
        while name in blocks:
            name += "_"
        new = _Block(name)
        new.term = [f"  br label {_ref(merge)}"]
        for i, line in enumerate(blocks[merge].phis):
            indent, res, ty, rest = _PHI.match(line).groups()
            entries = _PHI_ENTRY.findall(rest)
            moved = [(v, b) for v, b in entries if b[1:].strip('"') in inside]
            kept = [(v, b) for v, b in entries if b[1:].strip('"') not in inside]
            forwarded = f'%"{name}.phi{i}"'
            new.phis.append(
                f"{indent}{forwarded} = phi {ty} "
                + ", ".join(f"[ {v}, {b} ]" for v, b in moved)
            )
            kept.append((forwarded, _ref(name)))
            blocks[merge].phis[i] = f"{indent}{res} = phi {ty} " + ", ".join(
                f"[ {v}, {b} ]" for v, b in kept
            )
        for pred in inside:
            blocks[pred].retarget(merge, name)

        # Keep the bookkeeping current for the remaining headers.
        blocks[name] = new
        succs[name], preds[name] = [merge], list(inside)
        for pred in inside:
            succs[pred] = [name if s == merge else s for s in succs[pred]]
        preds[merge] = outside + [name]
        idom[name] = header
        depth[name] = depth[header] + 1
        ipdom[name] = merge
        loop_of[name] = loop_of.get(merge)
        changed = True
    return changed


def structurize(text):
    """Bring the control flow of every function into properly nested form.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The IR with unstructured joins duplicated and with forwarding merge
        blocks inserted where selections shared a merge block. Functions
        that are already structured are returned unchanged.
    """

    def rewrite(match):
        """Restructure the body of one function definition."""
        body = match.group(1)
        blocks = _parse(body)
        if not blocks:
            return match.group(0)
        unified = _unify_loop_exits(blocks)
        duplicated = _duplicate_joins(blocks)
        split = _split_merges(blocks)
        if not (unified or duplicated or split):
            return match.group(0)
        lines = []
        for block in blocks.values():
            lines += block.lines() + [""]
        return match.group(0).replace(body, "\n".join(lines))

    return _DEFINE.sub(rewrite, text)
