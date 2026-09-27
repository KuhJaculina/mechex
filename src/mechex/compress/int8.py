"""Integer post-training quantisation with per-channel scales.

Ref: Sec. 3.8, the int8 variant used post-training quantisation with per-channel scales and
was evaluated on the identical test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

INT8_MAX = 127


@dataclass(frozen=True)
class QuantisedLayer:
    """A quantised weight matrix with its per-channel scales."""

    name: str
    integer_weights: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]

    def dequantise(self) -> torch.Tensor:
        """Recover a floating-point weight from the integer representation."""
        return self.integer_weights.to(torch.float32) * self.scales.view(-1, 1)

    def error(self, reference: torch.Tensor) -> dict[str, float]:
        """Quantisation error against the reference weights."""
        recovered = self.dequantise()
        difference = (recovered - reference).abs()
        return {
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "relative_frobenius_error": float(
                torch.linalg.norm(recovered - reference)
                / torch.linalg.norm(reference).clamp_min(1e-12)
            ),
        }


def quantise_linear_per_channel(layer: nn.Linear, name: str = "") -> QuantisedLayer:
    """Quantise one linear layer with a scale per output channel.

    A per-tensor scale would be dominated by the largest row and crush the smaller ones;
    the per-channel scale is what the manuscript reports.
    """
    weight = layer.weight.detach().to(torch.float32)
    if weight.dim() != 2:
        raise ValueError("the layer weight must be a matrix")
    scales = weight.abs().amax(dim=1, keepdim=True) / INT8_MAX
    scales = scales.clamp_min(1e-12)
    integer = torch.round(weight / scales).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return QuantisedLayer(
        name=name or "linear",
        integer_weights=integer,
        scales=scales.squeeze(-1),
        original_shape=tuple(weight.shape),
    )


def quantise_module(module: nn.Module) -> dict[str, QuantisedLayer]:
    """Quantise every linear layer of a module, keyed by parameter path."""
    quantised: dict[str, QuantisedLayer] = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            quantised[name or "root"] = quantise_linear_per_channel(child, name or "root")
    return quantised


def quantisation_report(module: nn.Module) -> dict[str, object]:
    """Per-layer error of a post-training quantisation pass."""
    layers: list[dict[str, object]] = []
    total_error = 0.0
    count = 0
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        quantised = quantise_linear_per_channel(child, name or "root")
        error = quantised.error(child.weight.detach().to(torch.float32))
        layers.append({"layer": name or "root", **error})
        total_error += error["relative_frobenius_error"]
        count += 1
    return {
        "n_layers": count,
        "mean_relative_frobenius_error": total_error / max(count, 1),
        "layers": layers,
    }


def memory_footprint_bytes(module: nn.Module, bits: int = 8) -> dict[str, int]:
    """Parameter storage before and after quantisation."""
    float_bytes = 0
    quantised_bytes = 0
    for parameter in module.parameters():
        elements = parameter.numel()
        float_bytes += elements * 4
        scales = parameter.shape[0] if parameter.dim() >= 2 else 1
        quantised_bytes += elements * bits // 8 + scales * 4
    return {"float32_bytes": float_bytes, "quantised_bytes": quantised_bytes}


@torch.no_grad()
def evaluate_quantised_linear(
    layer: nn.Linear,
    inputs: torch.Tensor,
    quantised: QuantisedLayer | None = None,
) -> dict[str, float]:
    """Compare a quantised layer's output against the full-precision output.

    The quantised weight replaces the layer weight for this evaluation only; no layer in the
    release is mutated in place, so a quantisation experiment cannot silently change the
    model that was trained.
    """
    resolved = quantised or quantise_linear_per_channel(layer, "layer")
    reference = layer(inputs)
    recovered = torch.nn.functional.linear(
        inputs, resolved.dequantise(), layer.bias.detach() if layer.bias is not None else None
    )
    difference = (recovered - reference).abs()
    scale = reference.abs().mean().clamp_min(1e-12)
    return {
        "max_absolute_deviation": float(difference.max()),
        "mean_absolute_deviation": float(difference.mean()),
        "relative_deviation": float(difference.mean() / scale),
    }
