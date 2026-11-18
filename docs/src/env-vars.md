## Environment Variables

### RUNAI_STREAMER_CONCURRENCY

Controls the level of concurrency reading tensors into the CPU buffer.

This variable is now a fallback for both backends. It supplies `RUNAI_STREAMER_OBJ_CONCURRENCY` for
object storage, and `RUNAI_STREAMER_FS_QUEUE_DEPTH` for the file system, in each case only when that
variable is unset. 

#### Values accepted

Positive integer value, at most 1024. A larger value is capped, with a warning.

#### Default value

See `RUNAI_STREAMER_OBJ_CONCURRENCY` and `RUNAI_STREAMER_FS_QUEUE_DEPTH` - each backend keeps its own
default when nothing is set.

### RUNAI_STREAMER_OBJ_CONCURRENCY

Since version 0.17.0

Controls how much object-storage work runs at once.

Each unit is one client. For S3 a client targets 10 gigabits per second, so a concurrency of 8 targets
80 across the process.

When this variable is unset, `RUNAI_STREAMER_CONCURRENCY` supplies the value if it is set.

#### Values accepted

Positive integer value, at most 1024. A larger value is capped, with a warning.

#### Default value

8

### RUNAI_STREAMER_FS_QUEUE_DEPTH

Since version 0.17.0

Controls how many file system reads are in flight at once.

#### Values accepted

A positive integer, optionally followed by per-filesystem-type overrides:

```
RUNAI_STREAMER_FS_QUEUE_DEPTH=64                           # every mount
RUNAI_STREAMER_FS_QUEUE_DEPTH="64,nfs=32"                  # 32 on NFS, 64 elsewhere
RUNAI_STREAMER_FS_QUEUE_DEPTH="64,nfs=32,virtiofs=128"     # and 128 on virtiofs
```

The leading value is the default and is mandatory.
A type key matches as a prefix, so `nfs` covers both `nfs` and `nfs4`. Where two keys both match, the first one written wins.

The leading value is also the thread count for `sync_buffered`, and there it is capped at 1024. The
depth for the asynchronous readers is not capped, because a read in flight costs a queue slot rather
than a thread.

The type is the filesystem name reported by `/proc/self/mountinfo` for that mount - for example `ext4`,
`xfs`, `nfs`, `nfs4`, `virtiofs`, `overlay`. You can read the names on your own machine with:

```
findmnt -no FSTYPE /path/to/model
```

#### Default value

64 for the asynchronous readers

16 for `sync_buffered`

### RUNAI_STREAMER_FS_MAX_ENGINES

Since version 0.17.0

Controls how many asynchronous engines are built for each queue depth. An engine reads its mounts on
its own thread.

Every depth in `RUNAI_STREAMER_FS_QUEUE_DEPTH` gets an engine, so this variable is not needed to make
a per-filesystem-type depth work. Raise it to separate mounts that read at the same depth: above the
limit they share an engine, and a slow mount then delays the mounts sharing it.

Not used by `sync_buffered`, which serves all mounts from a single shared pool.

#### Values accepted

Positive integer, at most 1024. A larger value is capped, with a warning.

#### Default value

1

### RUNAI_STREAMER_FS_DIRECT_ASYNC

Whether the `io_uring` strategies hand reads to the kernel's io-wq workers (`IOSQE_ASYNC`) instead of
issuing them inline on the streamer's one I/O thread per mount. When on:

- **Block devices** (local NVMe, RAID, EBS and other block volumes): direct reads go to io-wq. This
  speeds up loads into newly allocated host memory.
- **NFS**: direct reads go to io-wq too. An inline direct read on NFS pins its destination and builds
  one request per page before any RPC goes out, so a deep window is read one request at a time. Each
  file is also read through several descriptors, and at most 8 reads of one file are in flight; the
  rest of the window goes to the next files queued, oldest first. A cold file server does not serve
  one file faster to more concurrent readers, so a whole window on one file only spends server CPU.
  The cap applies to `io_uring_buffered` reads on NFS as well, which always go to io-wq.

Measured with one process loading to host memory, against a disk-backed NFS export with server and
client caches dropped ("cold") or the server cache populated ("warm"):

| | inline (`0`) | `1` |
|---|---|---|
| 141 GB, 30 shards, kernel nfsd, NFSv4.2 over RDMA, cold | 51.1 s | 24.0 s |
| same, warm | 24.6 s | 6.6 s |
| same, NFS-Ganesha 5.9 limited to 16 CPUs, NFSv4.1 over TCP, cold | 116.3 s | 22.2 s |
| 61 GiB, 17 shards, NFS-Ganesha 4.3 limited to 32 CPUs, NFSv4.1 over TCP, cold | 12.9 s | 8.6 s |
| same, warm | 11.7 s | 5.8 s |
| same model as one 61 GiB file, cold | 12.4 s | 14.4 s |
| same, warm | 13.0 s | 6.1 s |

A checkpoint stored as one file has no other file to spread to, so on a cold server it can load about
15% slower than inline. Sharded checkpoints, and any warm server, are faster.

Set to `0` to issue every direct read inline and leave NFS reads unspread, as before. `libaio_direct`
is not affected.

#### Values accepted

Boolean: `1` or `0`

#### Default value

`1`

### RUNAI_STREAMER_FS_DIRECT_FILE_DEPTH

How many reads of one file on NFS may be in flight at once, under the `io_uring` strategies while
`RUNAI_STREAMER_FS_DIRECT_ASYNC` is on. The rest of the window goes to the next files queued, oldest
first. Has no effect on other file systems, under `libaio_direct`, or with `RUNAI_STREAMER_FS_DIRECT_ASYNC=0`.

The cap counts chunks, not bytes: the file server sees up to this many times
`RUNAI_STREAMER_FS_CHUNK_BYTESIZE` (8 MiB by default) divided by the mount's `rsize` reads of one file
at once. With kernel nfsd reading one file cold, about 4 MiB in flight per file was fastest, because
every page of a file beyond that contends on the same lock in the server's page cache. A server short
of CPU that serves single-file checkpoints cold may therefore do better with a lower cap and smaller
chunks. Raise the cap, or set `0`, for a fast warm server: a single file read at 15 GB/s with a cap of
4 and at 23 GB/s with 8.

#### Values accepted

Non-negative integer, at most 1024. `0` removes the cap. A larger value is capped, with a warning.

#### Default value

8

### RUNAI_STREAMER_FS_DIRECT_FDS

How many descriptors each file on NFS is opened with for direct reads, used in turn, under the
`io_uring` strategies while `RUNAI_STREAMER_FS_DIRECT_ASYNC` is on. The Linux NFS client keeps one lock
context per open file, so io-wq workers reading through one descriptor contend on it: a 141 GB load
over NFSv4.2/RDMA reached 16.9 GB/s through one descriptor and 20.9 GB/s through eight, at less than
a fifth of the CPU. Each file then holds this many descriptors while it is read; lower it if the
process is short of descriptors.

#### Values accepted

Integer from 1 to 64. A value outside that range is clamped, with a warning.

#### Default value

8

### RUNAI_STREAMER_CHUNK_BYTESIZE

Controls the maximum size of memory each OS thread reads from the file at once.

#### Values accepted

Positive integer value

> [!NOTE]
> When reading from file system the minimum size is 2097152 (=2MiB)
>
> When reading from object store the minimum size is 5242880 (=5MiB)

#### Default value

2097152 (=2MiB) when reading from file system

8388608 (=8MiB) when reading from object store e.g. S3

### RUNAI_STREAMER_MEMORY_LIMIT

Controls how the CPU Memory buffer to which tensors are read from the file is being limited. Read more about it [here](usage.md#cpu-memory-capping).

#### Values accepted

`-1 - UNLIMITED`, `0 - MIN`, or `Positive integer value - LIMITED`

#### Default value

40 GB. Under distributed streaming this is a node total, divided between the ranks on that node.

### RUNAI_STREAMER_RING_BUFFERS

How many buffers `RUNAI_STREAMER_MEMORY_LIMIT` is divided into when reading into host memory.

Each buffer is that memory divided by this number, but never smaller than the largest single tensor,
because a tensor cannot span two buffers. A model with large tensors therefore gets fewer buffers
than asked for.

This setting does not affect a read straight into a GPU. There the buffers are sized from the
tensors themselves - see `RUNAI_STREAMER_DEVICE_MEMORY_LIMIT`.

The buffer size matters beyond read-ahead here. Under the default
`owned=True` a yielded tensor is a view into its buffer and the whole buffer is handed to the caller,
so holding one tensor keeps its whole buffer alive - see
[Tensor ownership](usage.md#tensor-ownership-owned). More, smaller buffers lower peak memory for a
caller that retains tensors.

#### Values accepted

Positive integer value

#### Default value

`18`

### RUNAI_STREAMER_DEVICE_MEMORY_LIMIT

The most device memory the streamer may use for its buffers when it reads straight into a GPU.

A maximum, and only when it is set. The streamer does not aim for this number. It sizes its buffers
from the tensors it is about to read, and this setting and `RUNAI_STREAMER_DEVICE_MEMORY_FRACTION`
can only lower the result.

#### How the size is chosen

1. One buffer is the largest single tensor in the request, because a tensor cannot span two
   buffers. It is raised to 256 MiB if the largest tensor is smaller than that, and it is never
   larger than the whole request.
2. There are two buffers, so one can be filled while the other is in use. A request that fits in a
   single buffer gets one.
3. Under distributed streaming each rank adds up to two more buffers of the same size, to receive
   what its peers send.
4. That total is then capped by `RUNAI_STREAMER_DEVICE_MEMORY_FRACTION`, and by this setting when
   it is set. If the cap does not leave room, the buffers shrink, down to the largest tensor.

When even two of the largest tensor do not fit, the streamer reads into host memory and copies to
the device instead.

Under distributed streaming a rank that sends its buffers to its peers does not do that. It tries
once more with a single buffer of the same size. If that does not fit either, the load fails with an
error naming the device, the size it asked for and the free memory.

Unlike `RUNAI_STREAMER_MEMORY_LIMIT`, this setting has no unlimited or minimum mode.

#### Values accepted

Unset, or a positive integer value in bytes

#### Default value

Unset

### RUNAI_STREAMER_DEVICE_MEMORY_FRACTION

The share of a device's free memory the streamer may use for the buffers above.

It leaves the rest for the model being loaded.

#### Values accepted

A number greater than `0` and at most `1`

#### Default value

`0.5`

### RUNAI_STREAMER_ORDERED_YIELD

Whether tensors are handed back in file order - by the order of the files given to the streamer, and
by offset within each file - rather than in the order their reads completed.

On by default. Turning it off yields each tensor as soon as its read finishes.

> [!NOTE]
> This applies to a single process reading for itself (tensor parallel size 1). It does not yet
> affect the order under distributed streaming.

#### Values accepted

`0 - OFF`, or any other integer - ON

#### Default value

`1` (ordered)

### AWS_ENDPOINT_URL

Overrides url endpoint for reading from S3 compatible object store

> [!NOTE]
> 
> Mandatory for S3 compatible e.g. gcs, seaweedfs, unless [AWS_ENDPOINT_URL_S3](#aws_endpoint_url_s3) is set
> 
> Optional if reading from AWS S3

#### Values accepted

String

#### Default value

Default S3 url endpoint

### AWS_ENDPOINT_URL_S3

Overrides url endpoint for S3 only. Takes precedence over [AWS_ENDPOINT_URL](#aws_endpoint_url)

In the AWS SDKs `AWS_ENDPOINT_URL` applies to every service, while `AWS_ENDPOINT_URL_S3` applies to S3 alone. Use it when credentials are resolved through another AWS service, e.g. STS `AssumeRoleWithWebIdentity` (EKS IRSA) or AWS GovCloud FIPS endpoints, so that those calls are not sent to the S3 endpoint

> [!NOTE]
>
> An endpoint passed explicitly in the S3 credentials takes precedence over both variables
>
> An empty value is ignored

#### Values accepted

String

#### Default value

Value of `AWS_ENDPOINT_URL`, or the default S3 url endpoint

### AWS_CA_BUNDLE

Specifies the path to a certificate bundle to use for HTTPS certificate validation.

If defined, this environment variable overrides the value for the profile setting ca_bundle.

#### Values accepted

String

### RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING

Controls parsing the url endpoint for reading from object store 

> [!NOTE]
> Optional - No need to set this variable for reading from S3
> 
> Should be `0` for reading from compatible object store

#### Values accepted

Boolean `0` or `1`

#### Default value

`1`

### RUNAI_STREAMER_S3_UNSIGNED

Enables unsigned (anonymous) requests to S3. Use this when accessing public S3 buckets that do not require authentication.

#### Values accepted

Boolean `0` or `1`

#### Default value

`0`

### RUNAI_STREAMER_S3_TARGET_GBPS

Overrides the AWS CRT throughput target of each S3 client.

The value is per client, so the process targets it times `RUNAI_STREAMER_OBJ_CONCURRENCY`. To give a
single client the whole capacity, set the concurrency to `1` and this to the total you want.

#### Values accepted

Positive integer, in Gbps

#### Default value

The AWS CRT default of 10, per client

### RUNAI_STREAMER_S3_MAX_CONNECTIONS

Usage depends on object storage type

For S3 it caps the connections each client may open.
A resource guard, not a throughput control: the CRT already scales connections from the throughput target.
Set it only to bound resource usage, for example against a file descriptor limit.

For GCS it sets the per-client thread count directly.

For Azure it is ignored.

#### Values accepted

Positive integer

#### Default value

Unset - the AWS CRT chooses

### RUNAI_STREAMER_S3_MAX_RETRIES

Overrides the AWS S3 CRT retry limit for each ranged-read request. The value is
the maximum number of native retry attempts after the initial request. These
retries are completed by the AWS CRT before Run:ai receives the final result for
the request.

Setting this variable to `0` does not disable native retries; the AWS CRT treats
`0` as its default retry limit. Leave the variable unset to preserve the complete
AWS CRT default retry configuration.

#### Values accepted

Non-negative integer. Only a positive value overrides the AWS CRT retry limit.

#### Default value

AWS S3 CRT default retry policy

### RUNAI_STREAMER_S3_TIMEOUT

Controls the application-level retry window for each S3 chunk, in seconds. The
window starts when the chunk is first submitted to S3, so time spent waiting in
the worker queue before its first submission does not consume the retry window.

After the AWS CRT finishes its native retries and reports a retryable failure,
Run:ai schedules another attempt of only the failed chunk using exponential
backoff with full jitter. A new attempt is scheduled only when its complete
backoff ends before the chunk's deadline. The deadline does not cancel an S3
request that is already in flight.

Application retries cover transport failures, HTTP 5xx responses, HTTP 408 and
HTTP 429. Permanent client errors such as HTTP 400, 401, 403 and 404 are returned
immediately as `FileAccessError`.

Set this variable to `0`, or leave it unset, to disable application-level
retries. In that mode, the original `FileAccessError` is returned after the AWS
CRT retry policy finishes.

#### Values accepted

Non-negative integer, in seconds

#### Default value

`0` (application-level retries disabled)

### RUNAI_STREAMER_GCS_CREDENTIAL_FILE

Specifies the path to a credential file to use for GCS authentication.

If not defined (default) Application Default Credentials are used.

#### Values accepted

String

### AZURE_STORAGE_ACCOUNT_NAME

Azure Storage account name. Required for all Azure Blob Storage authentication methods.

#### Values accepted

String

#### Default value

None

### AZURE_STORAGE_SAS_TOKEN

Shared Access Signature (SAS) token for Azure Blob Storage authentication. Used with AZURE_STORAGE_ACCOUNT_NAME.

The value should be the query string portion of the SAS URI (with or without leading `?`), e.g. `sv=2021-08-06&ss=b&srt=co&sp=rl&se=...&sig=...`

#### Values accepted

String

#### Default value

None

### AZURE_STORAGE_ENDPOINT_SUFFIX

Azure Blob Storage endpoint suffix. Override for sovereign clouds (China, US Government) or Azure Stack.

#### Values accepted

String (e.g. `blob.core.chinacloudapi.cn`, `blob.core.usgovcloudapi.net`)

#### Default value

`blob.core.windows.net`

### RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED

> **Experimental** — This feature is under active development and may change in future releases.

Controls whether the Azure Blob cache provider is enabled. When a compatible cache provider package (e.g., `dacs-client`) is installed alongside `runai-model-streamer`, the streamer auto-discovers and loads it at runtime to accelerate model loading from Azure Blob Storage via a distributed cache.

Set to `0` to disable the cache provider entirely, even if the package is installed. This is the recommended way to disable caching in case of issues.

See [Azure Blob Cache Provider (Experimental)](usage.md#azure-blob-cache-provider-experimental) in the usage guide for details.

#### Values accepted

`0`, `1`, `auto`

#### Default value

Unset — cache is auto-enabled when a compatible cache provider package is installed

### RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB

Path to the Azure Blob cache-provider shared library. A nonempty value overrides the default ImageVolume path. An unset or empty value uses the default. Missing paths and non-regular files are ignored, and the streamer tries package auto-discovery instead. This path is not checked when caching is disabled.

Specify the full filesystem path; bare library names are not supported.

In auto mode, if no usable library is found, reads use the Azure SDK. Required mode (`RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED=1`) still reports an error if no usable cache library is found.

#### Values accepted

Shared library path

#### Default value

`/opt/cache-client/usr/local/lib/python3.10/dist-packages/py_dacs_client/libStorageDirect.so`

### RUNAI_STREAMER_DIST

Enables distributed streaming for multiple devices

`auto` enables distributed streaming only when the distributed backend that serves the target device is `nccl` (CUDA) or `xccl` (XPU).
A group initialized with a device-prefixed backend such as `cuda:nccl,cpu:gloo` or `xpu:xccl,cpu:gloo` qualifies for its CUDA or XPU devices.
For a CPU device that group uses `gloo`, so distributed streaming then needs `RUNAI_STREAMER_DIST=1`.

### Values accepted

String `0` or `1` or `auto`

#### Default value

`auto`

### RUNAI_STREAMER_DIST_GLOBAL

Enables global distributed streaming for multiple devices, dividing the workload between multiple nodes

If not defined (default) distributed streaming is local - dividing the workload between processes of the same node 

### Values accepted

String `0` or `1`

#### Default value

`0`

### RUNAI_STREAMER_GCS_USE_GRPC

Enables the gRPC transport for the GCS client, which utilizes direct connectivity for higher throughput and lower latency when running within Google Cloud.

If not defined (default) the GCS client uses the standard HTTP/JSON transport.

#### Values accepted

String `0` or `1`

#### Default value

`0`
