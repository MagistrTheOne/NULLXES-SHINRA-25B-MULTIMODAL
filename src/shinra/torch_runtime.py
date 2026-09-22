"""Checked import for model entry points; configuration/audit never import this."""

from .environment import require_pytorch

torch = require_pytorch()
nn = torch.nn
