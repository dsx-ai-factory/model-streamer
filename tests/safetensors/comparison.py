# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

def are_tensors_bitwise_identical(name, t1, t2):
    if t1.dtype != t2.dtype:
        return False, f"Tensor {name} has different dtype: {t1.dtype} vs {t2.dtype}"
    if t1.shape != t2.shape:
        return False, f"Tensor {name} has different shape: {t1.shape} vs {t2.shape}"
    
    t1 = t1.detach()
    t2 = t2.detach()

    # Get the size of a single element in bytes
    # This is the "safe" way to handle types PyTorch doesn't fully support yet
    element_size = t1.element_size()

    if t1.is_complex():                                                                                                                                                                                                                                                                                                                                                           
        # Compare real and imaginary parts separately                                                                                                                                                                                                                                                                                                                             
        v1_real, v1_imag = t1.real, t1.imag                                                                                                                                                                                                                                                                                                                                       
        v2_real, v2_imag = t2.real, t2.imag
        if torch.equal(v1_real, v2_real) and torch.equal(v1_imag, v2_imag):
            return True, f"Tensor {name} is bitwise identical"
        return False, f"Tensor {name} has different values"

    try:
        if element_size == 4: # Float32 / Int32
            v1, v2 = t1.view(torch.int32), t2.view(torch.int32)
        elif element_size == 2: # Float16 / BFloat16 / Int16
            v1, v2 = t1.view(torch.int16), t2.view(torch.int16)
        elif element_size == 8: # Float64 / Int64
            v1, v2 = t1.view(torch.int64), t2.view(torch.int64)
        elif element_size == 1: # All 8-bit types (FP8, E8M0, Int8, Uint8)
            v1, v2 = t1.view(torch.uint8), t2.view(torch.uint8)
        else:
            # Fallback for any unusual sizes (like packed 4-bit stored in larger blocks)
            v1, v2 = t1.view(torch.uint8), t2.view(torch.uint8)
    except RuntimeError:
        # If bitwise view fails (e.g. non-contiguous), 
        # fallback to standard equal after ensuring contiguity
        v1, v2 = t1.contiguous(), t2.contiguous()

    res = torch.equal(v1, v2)
    if res is False:
        return False, f"Tensor {name} has different values"
    return True, f"Tensor {name} is bitwise identical"

def tensor_maps_are_equal(first, second):
    if len(first.items()) != len(second.items()):
        return False, "Does not have the same number of tensors"
    
    for name, our_tensor in first.items():
        if name not in second:
            return False, f"Tensor {name} missing from second map"
            
        if not our_tensor.is_contiguous():
            # Force contiguous for comparison if needed, though strictly 
            # we should compare them as they are.
            our_tensor = our_tensor.contiguous()
        
        their_tensor = second[name]
        if not their_tensor.is_contiguous():
            their_tensor = their_tensor.contiguous()
        
        eq, err = are_tensors_bitwise_identical(name, our_tensor, their_tensor)
        if not eq:
            return False, err
    return True, "Tensors match successfully"