"""Numpy evaluation of Pollen-format ONNX actors (no env dependency, shared with .venv-ref tools)."""

from __future__ import annotations

import numpy as np


class OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.out = self.s.get_outputs()[0].name
        meta = self.s.get_modelmeta().custom_metadata_map
        self.meta = meta

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Exported with batch dim 1; run row-batched if the graph is static.
        try:
            return self.s.run([self.out], {self.inp: obs.astype(np.float32)})[0]
        except Exception:
            return np.concatenate([self.s.run([self.out], {self.inp: o[None]})[0] for o in obs.astype(np.float32)])


class NumpyOnnxMLP:
    """Evaluate the exported actor (Sub, Div, Gemm, Elu chain) in numpy, batched.

    Checked against onnxruntime row-by-row at construction.
    """

    def __init__(self, path):
        import onnx
        from onnx import numpy_helper

        g = onnx.load(str(path)).graph
        init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in g.initializer}
        consts = {}
        for n in g.node:
            if n.op_type == "Constant":
                consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t).astype(np.float64)
        self.vals = {**init, **consts}
        self.nodes = [n for n in g.node if n.op_type != "Constant"]
        self.inp = g.input[0].name
        self.out = g.output[0].name
        for n in self.nodes:
            if n.op_type not in ("Sub", "Div", "Gemm", "Elu", "Identity"):
                raise NotImplementedError(n.op_type)
        self.ref = OnnxPolicy(path)
        self.meta = self.ref.meta
        x = np.random.default_rng(0).standard_normal((8, 61)).astype(np.float32)
        a = self(x)
        b = np.concatenate([self.ref.s.run([self.ref.out], {self.ref.inp: r[None]})[0] for r in x])
        self.check_err = float(np.abs(a - b).max())
        assert self.check_err < 1e-4, self.check_err

    def __call__(self, obs):
        v = dict(self.vals)
        v[self.inp] = obs.astype(np.float64)
        for n in self.nodes:
            ins = [v[i] for i in n.input if i]
            at = {a.name: a for a in n.attribute}
            if n.op_type == "Sub":
                y = ins[0] - ins[1]
            elif n.op_type == "Div":
                y = ins[0] / ins[1]
            elif n.op_type == "Identity":
                y = ins[0]
            elif n.op_type == "Elu":
                al = at["alpha"].f if "alpha" in at else 1.0
                y = np.where(ins[0] > 0, ins[0], al * (np.exp(np.minimum(ins[0], 0)) - 1))
            else:  # Gemm
                A_, B_ = ins[0], ins[1]
                if "transA" in at and at["transA"].i:
                    A_ = A_.T
                if "transB" in at and at["transB"].i:
                    B_ = B_.T
                alpha = at["alpha"].f if "alpha" in at else 1.0
                beta = at["beta"].f if "beta" in at else 1.0
                y = alpha * A_ @ B_ + (beta * ins[2] if len(ins) > 2 else 0)
            v[n.output[0]] = y
        return v[self.out].astype(np.float32)
