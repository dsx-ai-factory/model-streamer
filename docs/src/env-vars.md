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

Whether `io_uring_direct` hands direct reads to the kernel's io-wq workers (`IOSQE_ASYNC`) on mounts
backed by a block device (local NVMe, RAID, EBS and other block volumes). This speeds up loads into
newly allocated host memory.

Set to `0` to issue every direct read inline, as before. Buffered reads and `libaio_direct` are not
affected.

#### Values accepted

Boolean: `1` or `0`

#### Default value

`1`

### RUNAI_STREAMER_FS_WORKERS_PER_MOUNT

Since version 0.18.0

Controls how many asynchronous readers serve each mount. A reader is a thread with its own queue.

One mount, one reader, is the default. Storage that answers many requests at once - a scale-out
filer, or a striped local array - can be read faster by several readers than by one, because a
single reader can only have its own queue's worth of requests outstanding.

The value is for the NODE, not for one process, and is divided between the streamer processes on
it the same way `RUNAI_STREAMER_FS_QUEUE_DEPTH` is. Eight readers over eight processes is one each;
over two processes it is four each.

The division drops a remainder. Each process gets the whole part of the share, so the mount is read
by that number times the processes on the node: twelve readers over eight processes is one each, and
the mount is read by eight, not twelve. A process never gets fewer than one reader, so asking for
fewer readers than there are processes also gives one each. The streamer warns whenever the mount
ends up with a number other than the one asked for.

The mount's queue depth from `RUNAI_STREAMER_FS_QUEUE_DEPTH` is SHARED between its readers, not
given to each: four readers on a depth of 64 run at 16 each. The division drops a remainder here
too, without a warning: four readers on a depth of 10 run at 2 each, so the mount runs at 8. The reads in flight for the mount are
then about what they were; they are issued from more threads. Each reader costs a thread and its
own staging memory, so raising this is not free.

A reader never runs below a depth of 1. Asking for more readers than the depth therefore puts more
reads in flight for the mount than the depth asks for: eight readers on a depth of 4 run at 1 each,
which is 8 in flight rather than 4.

Not used by `sync_buffered`, which serves all mounts from a single shared pool.

#### Values accepted

Positive integer, at most 64. A larger value is capped, with a warning. Zero is read as one.

#### Default value

1

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
   buffers. It is raised to `RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB` divided by the buffer count
   if the largest tensor is smaller than that, and it is never larger than the request divided by
   the same count.
2. The number of buffers is `RUNAI_STREAMER_DEVICE_RING_SLOTS`, so one can be filled while the
   others are in use. A request with too few tensors to fill them all gets fewer.
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

### RUNAI_STREAMER_DEVICE_RING_SLOTS

Since version 0.18.0

How many buffers the device ring above is built from.

The total is shared between them, so more buffers means smaller ones. A buffer is never smaller
than the largest range the caller asks for.

The streamer may use fewer than asked where the device cannot hold that many, where the request has
too few ranges to fill them, or where the value is above `1024`.

> [!NOTE]
> Under distributed streaming every rank must be given the same value. It decides the buffer size
> as well as the count, and the ranks exchange data in buffers of one size. A rank that cannot hold
> the agreed count still uses fewer.

#### Values accepted

A whole number of `1` or more

#### Default value

`4`

### RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB

Since version 0.18.0

The least the device buffers above may hold in total, in gibibytes.

`RUNAI_STREAMER_DEVICE_MEMORY_LIMIT` and `RUNAI_STREAMER_DEVICE_MEMORY_FRACTION` still apply after
it: a value larger than the device allows is reduced, not refused. A buffer is also never larger
than the request's share of the buffer count, so a request smaller than this value does not reach
it.

> [!NOTE]
> Under distributed streaming every rank must be given the same value. It decides the buffer size,
> which the ranks exchange data in and must agree on.

#### Values accepted

A whole number of gibibytes, `1` or more

#### Default value

`8`

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
