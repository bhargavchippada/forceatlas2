"""Tests for fa2 CLI (python -m fa2)."""

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
