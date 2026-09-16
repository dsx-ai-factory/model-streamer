from __future__ import annotations
import os
import torch
import struct
import json
from typing import List, Tuple, Optional, Any
from runai_model_streamer.distributed_streamer.distributed_streamer import (DistributedStreamer, FileChunks)
from runai_model_streamer.s3_utils.s3_utils import S3Credentials, is_s3_path, is_gs_path, is_azure_path

SAFETENSORS_DATA_OFFSETS_KEY = "data_offsets"
SAFETENSORS_NAME_KEY = "name"
SAFETENSORS_SHAPE_KEY = "shape"
SAFETENSORS_DTYPE_KEY = "dtype"
SAFETENSORS_HEADER_BUFFER_SIZE = 8

# Define a safety limit (100MB) for the header
MAX_HEADER_SIZE = 100 * 1024 * 1024 

LITTLE_ENDIAN_LONG_LONG_STRUCT_FORMAT = "<Q"

def get_safetensors_dtype_map() -> dict:
    safetensors_to_torch_dtype = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8":  torch.int8,
        "U8":  torch.uint8,
        "BOOL": torch.bool,
        "C64": torch.complex64,
    }

    # Add unsigned types if available (PyTorch >= 2.3.0)
    for st_name, torch_name in [("U64", "uint64"), ("U32", "uint32"), ("U16", "uint16")]:
        if hasattr(torch, torch_name):
            safetensors_to_torch_dtype[st_name] = getattr(torch, torch_name)

    # Experimental types with their PyTorch attribute names
    # Note: If a type is listed here but not available in the current PyTorch version,
    # it won't be added to the type map. If a file contains such a dtype (e.g., "F4"),
    # get_torch_dtype() will raise a clear ValueError: "Unsupported dtype 'F4'".
    # This is correct forward-compatible behavior - fail fast with a clear error message.
    _EXPERIMENTAL_ALIASES = {
        "F8_E4M3": ["float8_e4m3fn", "float8_e4m3fnuz"],
        "F8_E5M2": ["float8_e5m2", "float8_e5m2fnuz"],
        "F8_E8M0": ["float8_e8m0fnu", "float8_e8m0fnuz"],
        "F4":      ["float4_e2m1fn_x2"],  # Not yet in PyTorch (as of 2.5.1)
    }

    for st_type, torch_aliases in _EXPERIMENTAL_ALIASES.items():
        for alias in torch_aliases:
            if hasattr(torch, alias):
                safetensors_to_torch_dtype[st_type] = getattr(torch, alias)
                break

    return safetensors_to_torch_dtype

safetensors_to_torch_dtype = get_safetensors_dtype_map()


def _validate_physical_length(path: str, expected_total_bytes: int) -> None:
    """The header's own numbers can be internally self-consistent (no gap, no overlap, every
    tensor's shape matches its declared byte count) and still not match the actual file - too
    short (truncated) or too long (unexplained trailing bytes, including a gap before the FIRST
    tensor - see test_gap_before_the_first_tensor_raises).

    # TODO: object storage is skipped entirely - no cheap size probe is wired up here for
    # S3/GCS/Azure paths (would need a HEAD/stat call per path, e.g. via FileStreamer.list_files).
    # Truncation/bloat on those backends is only caught lazily, per-tensor, same as before this
    # check existed."""
    if is_s3_path(path) or is_gs_path(path) or is_azure_path(path):
        return
    actual_bytes = os.path.getsize(path)
    if actual_bytes < expected_total_bytes:
        raise ValueError(
            f"Corrupted File: '{path}' is truncated - header declares {expected_total_bytes} "
            f"bytes of tensor data but the file is only {actual_bytes} bytes."
        )
    if actual_bytes > expected_total_bytes:
        raise ValueError(
            f"Corrupted File: '{path}' has {actual_bytes} bytes but the header only declares "
            f"{expected_total_bytes} bytes of tensor data - {actual_bytes - expected_total_bytes} "
            f"extra trailing bytes."
        )


class SafetensorsMetadata:
    def __init__(self, blob: Any, offset: int) -> None:
        self.offset = offset
        self.tensors_metadata = []
        self.read_sizes = []

        for name, safetensor_metadata in blob.items():
            if name != "__metadata__":
                # Ensure required keys exist before processing
                if SAFETENSORS_DATA_OFFSETS_KEY not in safetensor_metadata:
                    raise ValueError(f"Corrupted Header: Tensor '{name}' missing {SAFETENSORS_DATA_OFFSETS_KEY}")
                
                tensor_metadata = SafetensorMetadata(name, safetensor_metadata)
                self.tensors_metadata.append(tensor_metadata)

        # Size breaks ties so a ZERO ELEMENT tensor sorts before a real tensor that starts at the same
        # offset - data_offsets [16, 16] and [16, 48] share a start. Without the tie-break the order comes
        # from the header's key order via the stable sort, and if the real tensor happened to come first
        # the gap check below would compute 16 + 32 > 16 and reject a perfectly good file as overlapping.
        self.tensors_metadata.sort(key=lambda x: (x.offsets.start, x.get_bytesize()))

        for i in range(len(self.tensors_metadata)):
            current_tensor = self.tensors_metadata[i]

            # Logical check for individual tensor offsets
            if current_tensor.offsets.start > current_tensor.offsets.end:
                raise ValueError(f"Corrupted Offset: Tensor '{current_tensor.name}' start ({current_tensor.offsets.start}) > end ({current_tensor.offsets.end})")
            
            required_size = current_tensor.get_bytesize()

            if i < len(self.tensors_metadata) - 1:
                current_start = self.tensors_metadata[i].offsets.start
                next_start = self.tensors_metadata[i + 1].offsets.start
                
                # Check for overlapping tensors
                # If tensors overlap, next_start - current_start < size, leading to data corruption.
                if current_start + required_size > next_start:
                    raise ValueError(f"Corrupted File: Tensor '{current_tensor.name}' overlaps with next tensor. (Ends at {current_start + required_size}, next starts at {next_start})")

                # Check for holes/gaps between tensors
                if current_start + required_size < next_start:
                     raise ValueError(f"Corrupted File: Gaps between tensors are not allowed. Tensor '{current_tensor.name}' ends at {current_start + required_size}, but next tensor starts at {next_start}.")

            self.read_sizes.append(required_size)

    @staticmethod
    def from_files(fs: DistributedStreamer, filenames: List[str], s3_credentials: Optional[S3Credentials]) -> List[SafetensorsMetadata]:
        # 1. Read the first 8 bytes (The Header Size)
        fs.stream_files([FileChunks.contiguous(i, filenames[i], 0, [SAFETENSORS_HEADER_BUFFER_SIZE]) for i in range(len(filenames))], s3_credentials, "cpu", False)
        
        header_sizes = {}
        
        # Wrap the first loop to catch truncation during size read
        try:
            for file_index, ready_chunk_index, buffer in fs.get_chunks():
                if buffer.nelement() < 8:
                    raise ValueError(f"File {filenames[file_index]} is truncated or empty.")

                size = struct.unpack(
                    LITTLE_ENDIAN_LONG_LONG_STRUCT_FORMAT, buffer.numpy()
                )[0]

                if size > MAX_HEADER_SIZE:
                     raise ValueError(f"Corrupted File: Header size {size} in {filenames[file_index]} exceeds limit of {MAX_HEADER_SIZE} bytes.")
                
                header_sizes[file_index] = size
        except Exception as e:
            # Catch the generic libstreamer error and re-raise as ValueError for consistency
            if isinstance(e, ValueError): raise e
            raise ValueError(f"Streamer failed to read header size (likely truncated file): {str(e)}")

        # 2. Read the JSON Header Body
        metadatas = {}
        fs.stream_files([FileChunks.contiguous(i, filenames[i], SAFETENSORS_HEADER_BUFFER_SIZE, [header_size]) for i, header_size in header_sizes.items()], s3_credentials, "cpu", False)
        
        # Wrap the second loop to catch truncation during JSON read
        try:
            for file_index, ready_chunk_index, buffer in fs.get_chunks():
                try:
                    json_bytes = bytearray(buffer.numpy())
                    json_str = json_bytes.decode('utf-8')
                    metadatas[file_index] = json.loads(json_str)
                except UnicodeDecodeError:
                    raise ValueError(f"Corrupted File: Header in {filenames[file_index]} is not valid UTF-8.")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Corrupted File: Header in {filenames[file_index]} is not valid JSON. Error: {e}")
        except Exception as e:
            if isinstance(e, ValueError): raise e
            raise ValueError(f"Streamer failed to read header body (likely truncated file): {str(e)}")

        results = []
        for i in range(len(filenames)):
            smeta = SafetensorsMetadata(metadatas[i], header_sizes[i] + SAFETENSORS_HEADER_BUFFER_SIZE)
            # Sum of declared sizes: the pairwise gap/overlap check above only ever compares
            # CONSECUTIVE tensors, so a gap between the data section's start and the FIRST tensor
            # is never checked anywhere else (see test_gap_before_the_first_tensor_raises).
            total_declared_bytes = sum(tm.get_bytesize() for tm in smeta.tensors_metadata)
            _validate_physical_length(filenames[i], smeta.offset + total_declared_bytes)
            results.append(smeta)
        return results

class SafetensorMetadata:
    def __init__(self, name: str, safetensorMetadata: Any) -> None:
        self.name = name
        self.shape = safetensorMetadata[SAFETENSORS_SHAPE_KEY]
        self.dtype = safetensorMetadata[SAFETENSORS_DTYPE_KEY]
        self.offsets = Offsets(safetensorMetadata[SAFETENSORS_DATA_OFFSETS_KEY])
        
        self._validate_shape_consistency()

    def _validate_shape_consistency(self):
        # 1. Calculate total number of elements
        num_elements = 1
        for dim in self.shape:
            num_elements *= dim
        
        # 2. Identify the actual bytes reserved in the file
        actual_bytes = self.offsets.get_diff()        

        torch_dtype = self.get_torch_dtype()
        element_size = torch.tensor([], dtype=torch_dtype).element_size()
        expected_bytes = num_elements * element_size

        # 3. Final Validation
        if expected_bytes != actual_bytes:
            raise ValueError(
                f"Corrupted Tensor '{self.name}': "
                f"Shape claims {expected_bytes} bytes ({self.shape} x {self.dtype}), "
                f"but offsets reserve {actual_bytes} bytes."
            )

    def get_bytesize(self) -> int:
        return self.offsets.get_diff()

    def get_item_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def get_torch_dtype(self) -> torch.dtype:
        # Handle unknown/unsupported dtypes
        if self.dtype not in safetensors_to_torch_dtype:
            raise ValueError(f"Unsupported dtype '{self.dtype}' in tensor '{self.name}'")
        return safetensors_to_torch_dtype[self.dtype]


class Offsets:
    def __init__(self, offsets: List[int]) -> None:
        self.start = offsets[0]
        self.end = offsets[1]

    def get_diff(self) -> int:
        return self.end - self.start


def prepare_request(
    fs: DistributedStreamer, paths: List[str], s3_credentials: Optional[S3Credentials]
) -> List[Tuple[int, List[SafetensorMetadata], List[int]]]:
    safetensors_metadatas = SafetensorsMetadata.from_files(fs, paths, s3_credentials)
    return [(
        safetensors_metadata.offset,
        safetensors_metadata.tensors_metadata,
        safetensors_metadata.read_sizes,
    ) for safetensors_metadata in safetensors_metadatas]


def create_torch_tensor(
    buffer: Any, tensor_metadata: SafetensorMetadata
) -> torch.Tensor:
    if tensor_metadata.get_item_count() == 0:
        return torch.empty(tensor_metadata.shape, dtype=tensor_metadata.get_torch_dtype())

    tensor = buffer.view(tensor_metadata.get_torch_dtype())
    
    # Reshape the tensor to its final, correct shape.
    return tensor.view(tensor_metadata.shape)