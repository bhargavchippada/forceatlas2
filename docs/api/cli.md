# CLI Reference

ForceAtlas2 provides a command-line interface via `python -m fa2`.

## Commands

### `layout` — Compute positions

```bash
python -m fa2 layout edges.json -o layout.json
python -m fa2 layout edges.csv --mode community --iterations 200 --seed 42
cat edges.json | python -m fa2 layout - > positions.json
```

### `render` — Layout and render to image

```bash
python -m fa2 render edges.json -o graph.png
python -m fa2 render edges.csv -o graph.svg --title "My Network" --mode community
```

### `metrics` — Compute layout quality

```bash
python -m fa2 metrics edges.json
python -m fa2 metrics edges.json --positions layout.json
```

## Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--iterations` | `-i` | 100 | Number of layout iterations |
| `--dim` | `-d` | 2 | Layout dimensions |
| `--mode` | `-m` | default | Layout preset: default, community, hub-dissuade, compact |
| `--seed` | `-s` | None | Random seed |
| `--output` | `-o` | stdout | Output file path |
| `--title` | `-t` | None | Chart title (render only) |
| `--positions` | `-p` | None | Pre-computed positions file (metrics only) |

## Input Formats

### JSON

```json
[["A", "B"], ["B", "C", 5.0]]
```

Or with explicit structure:

```json
{"edges": [["A", "B"], ["B", "C"]], "nodes": ["A", "B", "C"]}
```

Or adjacency dict:

```json
{"A": ["B", "C"], "B": ["A"]}
```

### CSV

```csv
source,target,weight
A,B,1.0
B,C,2.0
```

## Piping

```bash
# Stdin → layout
echo '[["A","B"],["B","C"],["A","C"]]' | python -m fa2 layout - > layout.json

# Stdin → render
echo '[["A","B"],["B","C"],["A","C"]]' | python -m fa2 render - -o graph.png

# Stdin → metrics (computes layout internally)
echo '[["A","B"],["B","C"],["A","C"]]' | python -m fa2 metrics -
```
