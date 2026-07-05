def selective_state_update(*a, **k):
    raise RuntimeError("selective_state_update shim: CUDA-only path; CPU must use torch_forward")
