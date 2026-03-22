"""CLI entry point for fa2.

Usage::

    python -m fa2 layout edges.json --mode community --output layout.json
    python -m fa2 render edges.csv --output graph.png
    cat edges.json | python -m fa2 layout > positions.json
"""

import argparse
import csv
import json
import sys


def _read_edges(source):
    """Read edges from file path or stdin. Supports JSON and CSV."""
    if source == "-" or source is None:
        text = sys.stdin.read()
    else:
        with open(source) as f:
            text = f.read()

    text = text.strip()
    if not text:
        return []

    # Try JSON first
    if text.startswith("[") or text.startswith("{"):
        data = json.loads(text)
        if isinstance(data, dict):
            if "edges" in data:
                return data["edges"]
            if "nodes" in data and "edges" not in data:
                return []  # graph with nodes only, no edges
            return data  # adjacency dict
        return data  # list of edges

    # Try CSV
    lines = text.splitlines()
    reader = csv.reader(lines)
    edges = []
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        if row[0].lower() in ("source", "src", "from", "node1"):
            continue
        if len(row) == 2:
            edges.append((_auto_type(row[0]), _auto_type(row[1])))
        elif len(row) >= 3:
            edges.append((_auto_type(row[0]), _auto_type(row[1]), float(row[2])))
    return edges


def _auto_type(val):
    """Convert string to int if possible, else keep as string."""
    try:
        return int(val)
    except ValueError:
        return val


def cmd_layout(args):
    """Run layout and output positions as JSON."""
    from fa2.easy import layout

    edges = _read_edges(args.input)
    positions = layout(
        edges,
        iterations=args.iterations,
        dim=args.dim,
        mode=args.mode,
        seed=args.seed,
    )

    result = {str(k): list(v) for k, v in positions.items()}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Layout written to {args.output}", file=sys.stderr)
    else:
        json.dump(result, sys.stdout, indent=2)
        print()


def cmd_render(args):
    """Run layout and render to image."""
    from fa2.easy import visualize

    edges = _read_edges(args.input)

    fmt = "png"
    if args.output and args.output.lower().endswith(".svg"):
        fmt = "svg"

    result = visualize(
        edges,
        iterations=args.iterations,
        dim=args.dim,
        mode=args.mode,
        seed=args.seed,
        output=fmt,
        path=args.output,
        title=args.title,
    )

    if args.output:
        print(f"Rendered to {args.output}", file=sys.stderr)
    elif result is not None:
        sys.stdout.buffer.write(result)


def cmd_metrics(args):
    """Compute layout quality metrics."""
    from fa2.easy import layout
    from fa2.metrics import edge_crossing_count, neighborhood_preservation, stress

    edges = _read_edges(args.input)

    if args.positions:
        with open(args.positions) as f:
            positions = json.load(f)
        positions = {_auto_type(k): tuple(v) for k, v in positions.items()}
    else:
        positions = layout(edges, iterations=args.iterations, mode=args.mode, seed=args.seed)

    from fa2.easy import _parse_edges
    node_list, G = _parse_edges(edges)

    n = len(node_list)
    result = {}
    result["stress"] = stress(G, positions)

    # Infer dimensionality from positions, not from args
    if positions:
        sample = next(iter(positions.values()))
        if len(sample) == 2:
            result["edge_crossings"] = edge_crossing_count(G, positions)

    k = min(10, n - 1) if n > 1 else 1
    result["neighborhood_preservation"] = neighborhood_preservation(G, positions, k=k)

    json.dump(result, sys.stdout, indent=2)
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="fa2",
        description="ForceAtlas2 graph layout — CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(sub):
        sub.add_argument("input", nargs="?", default="-",
                         help="Input file (JSON or CSV edge list). Use - for stdin.")
        sub.add_argument("-i", "--iterations", type=int, default=100)
        sub.add_argument("-d", "--dim", type=int, default=2)
        sub.add_argument("-m", "--mode", default="default",
                         choices=["default", "community", "hub-dissuade", "compact"])
        sub.add_argument("-s", "--seed", type=int, default=None)
        sub.add_argument("-o", "--output", default=None, help="Output file path")

    p_layout = subparsers.add_parser("layout", help="Compute layout positions (JSON output)")
    add_common_args(p_layout)
    p_layout.set_defaults(func=cmd_layout)

    p_render = subparsers.add_parser("render", help="Layout and render to image (PNG/SVG)")
    add_common_args(p_render)
    p_render.add_argument("-t", "--title", default=None, help="Chart title")
    p_render.set_defaults(func=cmd_render)

    p_metrics = subparsers.add_parser("metrics", help="Compute layout quality metrics")
    add_common_args(p_metrics)
    p_metrics.add_argument("-p", "--positions", default=None,
                           help="Positions JSON file (if omitted, computes layout first)")
    p_metrics.set_defaults(func=cmd_metrics)

    args = parser.parse_args()

    try:
        args.func(args)
    except (ValueError, ImportError, FileNotFoundError) as e:
        print(f"fa2 error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
