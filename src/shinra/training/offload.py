"""Explicit FP32 CPU-master AdamW with optional immediate gradient offload.

Single-process backend. FSDP/ZeRO own their sharded optimizer states separately.
"""

import torch


class CPUAdamW:
    def __init__(
        self,
        parameters,
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        gradient_offload=True,
        pin_memory=False,
    ):
        self.parameters = [p for p in parameters if p.requires_grad]
        if any(hasattr(p, "to_local") for p in self.parameters):
            raise ValueError("CPUAdamW cannot own DTensor/FSDP parameters")
        self.masters = []
        for p in self.parameters:
            master = p.detach().to(device="cpu", dtype=torch.float32, copy=True)
            if pin_memory:
                master = master.pin_memory()
            self.masters.append(master.requires_grad_(True))
        self.optimizer = torch.optim.AdamW(
            self.masters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=False
        )
        self.gradient_offload = gradient_offload
        self._collected = False
        self.handles = []
        if gradient_offload:
            for p, master in zip(self.parameters, self.masters):

                def transfer(param, target=master):
                    if param.grad is not None:
                        gradient = param.grad.detach().to(device="cpu", dtype=torch.float32, copy=True)
                        if target.grad is None:
                            target.grad = gradient
                        else:
                            target.grad.add_(gradient)
                        param.grad = None

                self.handles.append(p.register_post_accumulate_grad_hook(transfer))

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self._collected = False
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.parameters:
            p.grad = None

    def _collect(self):
        if not self.gradient_offload and not self._collected:
            for p, master in zip(self.parameters, self.masters):
                master.grad = None if p.grad is None else p.grad.detach().float().cpu()
            self._collected = True

    def clip_grad_norm(self, max_norm):
        self._collect()
        return torch.nn.utils.clip_grad_norm_(self.masters, max_norm, error_if_nonfinite=True)

    @torch.no_grad()
    def step(self):
        self._collect()
        self.optimizer.step()
        for p, master in zip(self.parameters, self.masters):
            p.copy_(master, non_blocking=False)
        self._collected = False

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "masters": [p.detach() for p in self.masters]}

    def load_state_dict(self, state):
        if len(state["masters"]) != len(self.masters):
            raise ValueError("Optimizer parameter count mismatch")
        self.optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            for p, master, saved in zip(self.parameters, self.masters, state["masters"]):
                master.copy_(saved)
                p.copy_(master)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
