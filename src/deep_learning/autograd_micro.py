"""Scalar reverse-mode autograd in ~80 lines (a la Karpathy's micrograd).

Interview goal: be able to *derive* backprop. We build a Value class
that tracks a computation graph, then implement add/mul/relu/tanh and a
topological-sort backward pass.

We then train a 1-hidden-layer MLP on a tiny dataset to verify it works,
and cross-check the gradient against torch.autograd.
"""

from __future__ import annotations

import math
import random


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _bw():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _bw():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _bw
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _bw():
            self.grad += (1 - t * t) * out.grad
        out._backward = _bw
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _bw():
            self.grad += (out.data > 0) * out.grad
        out._backward = _bw
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        # Topological order so each node is processed after all its children.
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        s = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return s.tanh()

    def parameters(self):
        return self.w + [self.b]


class MLP:
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append([Neuron(sizes[i]) for _ in range(sizes[i + 1])])

    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x

    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]


def main():
    random.seed(0)
    # XOR-like task — classic non-linear separability check.
    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [-1.0, 1.0, 1.0, -1.0]
    net = MLP([2, 4, 1])
    print(f"Params: {len(net.parameters())}")

    for step in range(100):
        preds = [net(x)[0] for x in xs]
        loss = sum((p - y) * (p - y) for p, y in zip(preds, ys))
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in net.parameters():
            p.data -= 0.1 * p.grad
        if step % 20 == 0:
            print(f"step {step:3d}  loss {loss.data:.4f}")

    final = [round(net(x)[0].data, 2) for x in xs]
    print(f"Final preds: {final}  (target {ys})")

    # Cross-check one gradient against torch if available.
    try:
        import torch
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(-3.0, requires_grad=True)
        (a * b + a.tanh()).backward()
        am = Value(2.0); bm = Value(-3.0)
        (am * bm + am.tanh()).backward()
        print(f"\nGrad check vs torch:")
        print(f"  a: micro={am.grad:.4f}  torch={a.grad.item():.4f}")
        print(f"  b: micro={bm.grad:.4f}  torch={b.grad.item():.4f}")
        assert abs(am.grad - a.grad.item()) < 1e-6
    except ImportError:
        pass


if __name__ == "__main__":
    main()
