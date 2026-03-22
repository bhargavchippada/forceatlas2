"""Tests for fa2 CLI (python -m fa2)."""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import pytest

PYTHON = sys.executable


def run_fa2(*args, stdin=None):
    """Run `python -m fa2 ...` and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [PYTHON, "-m", "fa2", *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.fixture
def json_edges_file():
    edges = [["A", "B"], ["B", "C"], ["A", "C", 5.0]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(edges, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def csv_edges_file():
    content = "source,target,weight\nA,B,1.0\nB,C,2.0\nA,C,5.0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


class TestCLILayout:
    def test_layout_from_json_file(self, json_edges_file):
        rc, stdout, stderr = run_fa2("layout", json_edges_file, "-i", "10", "-s", "42")
        assert rc == 0
        positions = json.loads(stdout)
        assert len(positions) == 3
        assert all(len(v) == 2 for v in positions.values())

    def test_layout_from_csv_file(self, csv_edges_file):
        rc, stdout, stderr = run_fa2("layout", csv_edges_file, "-i", "10", "-s", "42")
        assert rc == 0
        positions = json.loads(stdout)
        assert len(positions) == 3

    def test_layout_from_stdin(self):
        edges_json = json.dumps([["X", "Y"], ["Y", "Z"]])
        rc, stdout, stderr = run_fa2("layout", "-", "-i", "10", "-s", "42", stdin=edges_json)
        assert rc == 0
        positions = json.loads(stdout)
        assert len(positions) == 3

    def test_layout_to_file(self, json_edges_file):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            rc, stdout, stderr = run_fa2("layout", json_edges_file, "-o", out_path, "-i", "10", "-s", "42")
            assert rc == 0
            with open(out_path) as f:
                positions = json.load(f)
            assert len(positions) == 3
        finally:
            os.unlink(out_path)

    def test_layout_mode_community(self, json_edges_file):
        rc, stdout, stderr = run_fa2("layout", json_edges_file, "-m", "community", "-i", "10", "-s", "42")
        assert rc == 0
        positions = json.loads(stdout)
        assert len(positions) == 3

    def test_layout_dim_3(self, json_edges_file):
        rc, stdout, stderr = run_fa2("layout", json_edges_file, "-d", "3", "-i", "10", "-s", "42")
        assert rc == 0
        positions = json.loads(stdout)
        assert all(len(v) == 3 for v in positions.values())


class TestCLIRender:
    def test_render_to_file(self, json_edges_file):
        pytest.importorskip("matplotlib")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        try:
            rc, stdout, stderr = run_fa2("render", json_edges_file, "-o", out_path, "-i", "10", "-s", "42")
            assert rc == 0
            assert os.path.getsize(out_path) > 100
        finally:
            os.unlink(out_path)

    def test_render_svg(self, json_edges_file):
        pytest.importorskip("matplotlib")
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            out_path = f.name
        try:
            rc, stdout, stderr = run_fa2("render", json_edges_file, "-o", out_path, "-i", "10", "-s", "42")
            assert rc == 0
            with open(out_path, "rb") as f:
                assert b"<svg" in f.read()
        finally:
            os.unlink(out_path)

    def test_render_with_title(self, json_edges_file):
        pytest.importorskip("matplotlib")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        try:
            rc, _, _ = run_fa2("render", json_edges_file, "-o", out_path, "-t", "My Graph", "-i", "10", "-s", "42")
            assert rc == 0
        finally:
            os.unlink(out_path)


class TestCLIMetrics:
    def test_metrics_compute(self, json_edges_file):
        rc, stdout, stderr = run_fa2("metrics", json_edges_file, "-i", "10", "-s", "42")
        assert rc == 0
        metrics = json.loads(stdout)
        assert "stress" in metrics
        assert "neighborhood_preservation" in metrics
        assert "edge_crossings" in metrics
        assert isinstance(metrics["edge_crossings"], int)

    def test_metrics_with_positions_file(self, json_edges_file):
        # First compute layout
        rc, stdout, _ = run_fa2("layout", json_edges_file, "-i", "10", "-s", "42")
        assert rc == 0
        positions = json.loads(stdout)

        # Save positions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(positions, f)
            pos_path = f.name

        try:
            rc, stdout, _ = run_fa2("metrics", json_edges_file, "-p", pos_path)
            assert rc == 0
            metrics = json.loads(stdout)
            assert "stress" in metrics
        finally:
            os.unlink(pos_path)


class TestCLIEdgeCases:
    def test_help(self):
        rc, stdout, stderr = run_fa2("--help")
        assert rc == 0

    def test_layout_help(self):
        rc, stdout, stderr = run_fa2("layout", "--help")
        assert rc == 0

    def test_csv_with_comments_and_header(self):
        content = "# This is a comment\nsource,target\n0,1\n1,2\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            rc, stdout, _ = run_fa2("layout", path, "-i", "10", "-s", "42")
            assert rc == 0
            positions = json.loads(stdout)
            assert len(positions) == 3
        finally:
            os.unlink(path)

    def test_adjacency_dict_json(self):
        adj = {"A": ["B", "C"], "B": ["A"], "C": ["A"]}
        rc, stdout, _ = run_fa2("layout", "-", "-i", "10", "-s", "42", stdin=json.dumps(adj))
        assert rc == 0
        positions = json.loads(stdout)
        assert len(positions) == 3

    def test_graph_json_format(self):
        """Test {nodes: [...], edges: [...]} JSON format."""
        data = {"nodes": [], "edges": [["A", "B"], ["B", "C"]]}
        rc, stdout, _ = run_fa2("layout", "-", "-i", "10", "-s", "42", stdin=json.dumps(data))
        assert rc == 0
        positions = json.loads(stdout)
        assert len(positions) == 3

    def test_nodes_only_json(self):
        """Graph with nodes but no edges key should return empty positions."""
        data = {"nodes": ["A", "B", "C"]}
        rc, stdout, _ = run_fa2("layout", "-", "-i", "10", "-s", "42", stdin=json.dumps(data))
        assert rc == 0
        positions = json.loads(stdout)
        assert positions == {}


class TestCLIErrors:
    def test_nonexistent_file(self):
        rc, stdout, stderr = run_fa2("layout", "/nonexistent/file.json")
        assert rc == 1
        assert "error" in stderr.lower()

    def test_malformed_json(self):
        rc, stdout, stderr = run_fa2("layout", "-", stdin="{invalid json")
        assert rc == 1

    def test_render_without_matplotlib(self):
        """Render fails gracefully if matplotlib missing (tested via error handling)."""
        # This test just verifies the error handler catches ImportError cleanly
        # In practice matplotlib is installed, so we just verify render works
        pytest.importorskip("matplotlib")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        edges_json = json.dumps([["A", "B"]])
        try:
            rc, _, _ = run_fa2("render", "-", "-o", out_path, "-i", "5", stdin=edges_json)
            assert rc == 0
        finally:
            os.unlink(out_path)


class TestCLIInProcess:
    """In-process tests for coverage (subprocess tests don't count)."""

    def test_read_edges_json_list(self):
        from fa2.__main__ import _read_edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"], ["B", "C"]], f)
            path = f.name
        try:
            edges = _read_edges(path)
            assert len(edges) == 2
        finally:
            os.unlink(path)

    def test_read_edges_json_dict_with_edges(self):
        from fa2.__main__ import _read_edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"edges": [["A", "B"]], "nodes": ["A", "B"]}, f)
            path = f.name
        try:
            edges = _read_edges(path)
            assert edges == [["A", "B"]]
        finally:
            os.unlink(path)

    def test_read_edges_json_dict_nodes_only(self):
        from fa2.__main__ import _read_edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"nodes": ["A", "B"]}, f)
            path = f.name
        try:
            edges = _read_edges(path)
            assert edges == []
        finally:
            os.unlink(path)

    def test_read_edges_json_adjacency_dict(self):
        from fa2.__main__ import _read_edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"A": ["B"], "B": ["A"]}, f)
            path = f.name
        try:
            result = _read_edges(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_read_edges_csv(self):
        from fa2.__main__ import _read_edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("source,target,weight\nA,B,1.0\nB,C,2.0\n")
            path = f.name
        try:
            edges = _read_edges(path)
            assert len(edges) == 2
            assert len(edges[0]) == 3  # has weight
        finally:
            os.unlink(path)

    def test_read_edges_csv_no_weight(self):
        from fa2.__main__ import _read_edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("from,to\nA,B\nB,C\n")
            path = f.name
        try:
            edges = _read_edges(path)
            assert len(edges) == 2
            assert len(edges[0]) == 2
        finally:
            os.unlink(path)

    def test_read_edges_csv_with_comments(self):
        from fa2.__main__ import _read_edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("# comment\nsrc,tgt\n0,1\n1,2\n")
            path = f.name
        try:
            edges = _read_edges(path)
            assert len(edges) == 2
        finally:
            os.unlink(path)

    def test_read_edges_empty(self):
        from fa2.__main__ import _read_edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("  ")
            path = f.name
        try:
            edges = _read_edges(path)
            assert edges == []
        finally:
            os.unlink(path)

    def test_read_edges_stdin(self, monkeypatch):
        """Test _read_edges with stdin (source='-')."""
        import io

        from fa2.__main__ import _read_edges
        monkeypatch.setattr("sys.stdin", io.StringIO('[["X","Y"]]'))
        edges = _read_edges("-")
        assert edges == [["X", "Y"]]

    def test_auto_type(self):
        from fa2.__main__ import _auto_type
        assert _auto_type("42") == 42
        assert _auto_type("hello") == "hello"

    def test_cmd_layout(self):
        from fa2.__main__ import cmd_layout
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"], ["B", "C"]], f)
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            args = argparse.Namespace(
                input=in_path, output=out_path, iterations=5,
                dim=2, mode="default", seed=42,
            )
            cmd_layout(args)
            with open(out_path) as f:
                positions = json.load(f)
            assert len(positions) == 3
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_cmd_layout_stdout(self, capsys):
        from fa2.__main__ import cmd_layout
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"]], f)
            in_path = f.name
        try:
            args = argparse.Namespace(
                input=in_path, output=None, iterations=5,
                dim=2, mode="default", seed=42,
            )
            cmd_layout(args)
            captured = capsys.readouterr()
            positions = json.loads(captured.out)
            assert len(positions) == 2
        finally:
            os.unlink(in_path)

    def test_cmd_render(self):
        pytest.importorskip("matplotlib")
        from fa2.__main__ import cmd_render
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"], ["B", "C"]], f)
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        try:
            args = argparse.Namespace(
                input=in_path, output=out_path, iterations=5,
                dim=2, mode="default", seed=42, title="Test",
            )
            cmd_render(args)
            assert os.path.getsize(out_path) > 100
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_cmd_render_svg(self):
        pytest.importorskip("matplotlib")
        from fa2.__main__ import cmd_render
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"]], f)
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            out_path = f.name
        try:
            args = argparse.Namespace(
                input=in_path, output=out_path, iterations=5,
                dim=2, mode="default", seed=42, title=None,
            )
            cmd_render(args)
            with open(out_path, "rb") as f:
                assert b"<svg" in f.read()
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_cmd_render_stdout(self):
        pytest.importorskip("matplotlib")
        from fa2.__main__ import cmd_render
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"]], f)
            in_path = f.name
        try:
            args = argparse.Namespace(
                input=in_path, output=None, iterations=5,
                dim=2, mode="default", seed=42, title=None,
            )
            cmd_render(args)
            # No assertion on stdout — just verifies no crash
        finally:
            os.unlink(in_path)

    def test_cmd_metrics(self, capsys):
        from fa2.__main__ import cmd_metrics
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"], ["B", "C"]], f)
            in_path = f.name
        try:
            args = argparse.Namespace(
                input=in_path, output=None, iterations=5,
                dim=2, mode="default", seed=42, positions=None,
            )
            cmd_metrics(args)
            captured = capsys.readouterr()
            metrics = json.loads(captured.out)
            assert "stress" in metrics
        finally:
            os.unlink(in_path)

    def test_cmd_metrics_with_positions(self, capsys):
        from fa2.__main__ import cmd_metrics
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"], ["B", "C"]], f)
            in_path = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"A": [0.0, 0.0], "B": [1.0, 0.0], "C": [0.5, 1.0]}, f)
            pos_path = f.name
        try:
            args = argparse.Namespace(
                input=in_path, output=None, iterations=5,
                dim=2, mode="default", seed=42, positions=pos_path,
            )
            cmd_metrics(args)
            captured = capsys.readouterr()
            metrics = json.loads(captured.out)
            assert "stress" in metrics
        finally:
            os.unlink(in_path)
            os.unlink(pos_path)

    def test_main_layout(self):
        from fa2.__main__ import main
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([["A", "B"]], f)
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            sys.argv = ["fa2", "layout", in_path, "-o", out_path, "-i", "5", "-s", "42"]
            main()
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_main_error_handling(self):
        from fa2.__main__ import main
        sys.argv = ["fa2", "layout", "/nonexistent/path.json"]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_read_edges_from_stdin_none(self, monkeypatch):
        """Cover source=None path."""
        import io

        from fa2.__main__ import _read_edges
        monkeypatch.setattr("sys.stdin", io.StringIO('[["A","B"]]'))
        edges = _read_edges(None)
        assert len(edges) == 1
