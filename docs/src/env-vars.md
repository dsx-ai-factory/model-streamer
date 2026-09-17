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
RUNAI_STREAMER_FS_QUEUE_DEPTH=512                          # every mount
RUNAI_STREAMER_FS_QUEUE_DEPTH="512,nfs=64"                 # 64 on NFS, 512 elsewhere
RUNAI_STREAMER_FS_QUEUE_DEPTH="512,nfs=64,virtiofs=256"    # and 256 on virtiofs
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

512 for the asynchronous readers

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

`-1` for distributed streaming and 40 GB otherwise

### AWS_ENDPOINT_URL

Overrides url endpoint for reading from S3 compatible object store

> [!NOTE]
> 
> Mandatory for S3 compatible e.g. gcs, minio
> 
> Optional if reading from AWS S3

#### Values accepted

String

#### Default value

Default S3 url endpoint

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

### RUNAI_STREAMER_DIST

Enables distributed streaming for multiple devices

`auto` enables distributed streaming only when using the `nccl` distributed backend

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
