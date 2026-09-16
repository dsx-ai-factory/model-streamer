## Using Model Streamer

### Streaming

Start streaming models by creating a `SafetensorsStreamer` object that will serve as a context manager. The `SafetensorsStreamer` object opens OS threads that will read the tensors from the Safetensor file to the CPU memory. Closing the object destroys the threads. You can control the number of concurrent threads with the environment variable `RUNAI_STREAMER_CONCURRENCY`.

#### Streaming from a file system

If your SafeTensors file resides on a file system, run the following code to load the tensors to the CPU buffer and stream them to the GPU memory:

```python
from runai_model_streamer import SafetensorsStreamer

file_path = "/path/to/file.safetensors"

with SafetensorsStreamer() as streamer:
    streamer.stream_file(file_path)
    for name, tensor in streamer.get_tensors():
        tensor.to('CUDA:0')
```

> **Note:** To make the tensors available on the CPU memory, clone the yielded tensors before calling `streamer.get_tensors()`. Note that otherwise, tensors may be overwritten when using `RUNAI_STREAMER_MEMORY_LIMIT` or completely destroyed when closing the `SafetensorsStreamer` object.

#### Streaming from multiple files

To stream tensors from multiple files in parallel use the `streamer.stream_files()` API:

```python
from runai_model_streamer import SafetensorsStreamer

file_paths = ["/path/to/file-1.safetensors", "/path/to/file-2.safetensors"]

with SafetensorsStreamer() as streamer:
    streamer.stream_files(file_paths)
    for name, tensor in streamer.get_tensors():
        tensor.to('CUDA:0')
```

> **Note:** You can not mix S3 path and file system paths on same `streamer.stream_files()` call.

#### Distributed streaming

##### Use case and motivation

Distributed streaming is for multiple processes which are reading the same file list, e.g., the default loader of vLLM loading model weights on multiple devices.

When reading from a file system, the operating system page cache optimizes the reading by storing pages in the cache, so files are read from storage only once. However, reading from object storage cannot utilize the page cache, leading to multiple reads from storage and long loading times.

Distributed streaming is designed to solve this problem by dividing the reading workload between the multiple processes, where each process reads a unique portion of the files and then distributes its share to the other processes.

##### Usage

```python
from runai_model_streamer import SafetensorsStreamer

file_paths = ["/path/to/file-1.safetensors", "/path/to/file-2.safetensors"]

tensors = {}
device = 'CUDA:0'
with SafetensorsStreamer() as streamer:
    streamer.stream_files(file_paths, s3_credentials=None, device=device, is_distributed=True)
    for name, tensor in streamer.get_tensors():       
       tensors[name] = tensor.clone().detach() # returning tensors on the specified device, which is CUDA:0
```

##### Requirements

Distributed streaming allocates reusable staging buffers on each device, which hold the data of the yielded tensors
Therefore, the yielded tensor might be overwritten at the next iteration. If tensors are used outside the iterator loop, clone and detach the yielded tensor to save a copy.
 
The memory requirements for the staging buffers is twice the size of the largest tensor in the files

Distributed streaming is based on a torch distributed group and the broadcast operation.
The backend of the torch group must support the broadcast operation.

The performance gain depends on the type of backend and the communication between devices.
The nccl backend with nvlink between devices is most suitable for distributed streaming.

##### Control

Distributed streaming is enabled by default when streaming from object storage to CUDA devices.
It is possible to disable distributed streaming by setting `RUNAI_STREAMER_DIST=0`

It is possible to force distributed streaming for other cases by setting `RUNAI_STREAMER_DIST=1`

##### Global and local modes

In local mode, the processes on each node divide the entire workload among themselves.

In global mode, the workload is divided between all the processes on all the nodes.

For example, with 2 nodes and 16 processes (8 processes on each node), each process reads 1/8 of the workload in local mode and 1/16 in global mode.

The mode should be selected according to the communication speed between the nodes.
By default, distributed streaming is done in local mode.
To enable global mode, set `RUNAI_STREAMER_DIST_GLOBAL=1`.

#### Streaming from S3

> **Note:** Streaming models from S3 requires the installation of the streamer S3 package, as can be found [here](#s3CapabilityInstallation).

To load tensors from object storage, replace the file path in the code above with your S3 path, e.g.:

```python
file_path = "s3://my-bucket/my/file/path.safetensors"
```

#### S3 authentication

###### Authentication via AWS CPP SDK (default)

By default, the S3 client in `libstreamers3.so` authenticates itself using the AWS C++ SDK's default credential provider chain. No boto3 session is created.

Credentials are resolved from any of the following:

1. Environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

2. Credentials file `~/.aws/credentials`

3. Temporary credentials generated by AWS Security Token Service (STS) and passed as environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`

If IAM role assumption is needed, session token should be created using the AWS Security Token Service (STS).

e.g. `aws sts assume-role --role-arn arn:aws:iam::<account_name>:role/<role> --role-session-name ecs-session`

The session token shoud be passed as an environment variable `AWS_SESSION_TOKEN`

To check if IAM role assumption is needed run `aws s3 ls s3://your-bucket-name --region your-region`. If you get a `403 Forbidden` error, you might need an assumed role

###### Authentication via boto3

Setting `RUNAI_STREAMER_NO_BOTO3_SESSION=0` makes the streamer resolve credentials via a boto3 session in Python and pass the resolved credentials to the S3 client in `libstreamers3.so`. This is useful when credentials are only reachable through boto3 (for example an SSO or `credential_process` profile).

###### HTTPS certificate

Custom certificates file can be passed using `AWS_CA_BUNDLE=path/to/ca_file`

The file path can also be configured in the `~/.aws/config` file with `ca_bundle = /path/to/ca_file`. This is honored in both authentication modes.

###### Unsigned requests (public buckets)

To access public S3 buckets that do not require authentication, set:

```bash
export RUNAI_STREAMER_S3_UNSIGNED=1
```

This configures the boto3 client to send unsigned (anonymous) requests, bypassing credential resolution entirely.

#### Streaming from Azure Blob Storage

> **Note:** Streaming models from Azure Blob Storage requires the installation of the streamer Azure package, as can be found [here](#azureCapabilityInstallation).

To load tensors from Azure Blob Storage, replace the file path in the code above with your Azure path, e.g.:

```python
file_path = "az://my-container/my/file/path.safetensors"
```

##### Azure Authentication

The streamer supports multiple authentication methods for Azure Blob Storage, checked in this order:

1. **SAS token** (`AZURE_STORAGE_SAS_TOKEN`)
2. **Storage account key** (`AZURE_STORAGE_ACCOUNT_KEY`)
3. **DefaultAzureCredential** (Recommended) — managed identity, Azure CLI, service principal, etc.

###### Default Azure Credential (Recommended)

Set the storage account name and DefaultAzureCredential handles authentication automatically:

```bash
export AZURE_STORAGE_ACCOUNT_NAME="myaccount"
```

The DefaultAzureCredential chain tries multiple authentication methods in order:
1. **Environment variables** (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`) - for service principal authentication
2. **Managed Identity** - no configuration needed when running in Azure (VMs, AKS, App Service, etc.)
3. **Azure CLI** - authenticate via `az login`
4. **Azure PowerShell** - authenticate via `Connect-AzAccount`
5. **Azure Developer CLI** - authenticate via `azd auth login`

See [DefaultAzureCredential](https://learn.microsoft.com/en-us/dotnet/azure/sdk/authentication#defaultazurecredential) for more information.

###### Service Principal Authentication

For automated pipelines and CI/CD, use service principal credentials:

```bash
export AZURE_STORAGE_ACCOUNT_NAME="myaccount"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_SECRET="your-client-secret"
```

###### Managed Identity

When running in Azure (VMs, AKS, Azure Functions, etc.), managed identity is used automatically:

```bash
export AZURE_STORAGE_ACCOUNT_NAME="myaccount"
# No additional configuration needed - managed identity is detected automatically
```

###### SAS Token

To authenticate using a Shared Access Signature:

```bash
export AZURE_STORAGE_ACCOUNT_NAME="myaccount"
export AZURE_STORAGE_SAS_TOKEN="sv=2021-08-06&ss=b&srt=co&sp=rl&se=2026-01-01T00:00:00Z&sig=..."
```

> **Note:** The SAS token value should be the query string portion of the SAS URI without the leading `?`.

##### Azure Blob Cache Provider (Experimental)

> **Experimental** — This feature is under active development and may change in future releases.

The streamer supports pluggable cache providers for Azure Blob Storage. When a compatible cache provider package is installed (e.g., `tachyon-client`), it is auto-discovered and loaded at runtime. All blob reads are then routed through the cache provider instead of the Azure SDK, enabling integration with distributed caches to accelerate model loading.

###### How it works

1. Install the cache provider package alongside `runai-model-streamer` (e.g., `pip install tachyon-client`)
2. At startup, the streamer auto-discovers the cache library in Python site-packages via `dladdr`
3. The library is loaded via `dlopen` and the `blob_read` symbol is resolved
4. All subsequent Azure blob reads are served through the cache provider
5. If no cache provider is installed, reads go directly to Azure Blob Storage — no regression

###### Disabling the cache

To disable the cache provider even when it is installed, set:

```bash
export RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED=0
```

This is the recommended way to disable caching in case of issues.

###### Implementing a cache provider

A cache provider is a shared library that exports a single C function:

```c
#include <stddef.h>
#include <sys/types.h>

extern "C" ssize_t blob_read(
    const char* account,      /* Azure Storage account name */
    const char* container,    /* Azure container name */
    const char* blob,         /* Blob path within the container */
    void* buf,                /* Output buffer (caller-allocated, >= length bytes) */
    size_t offset,            /* Byte offset within the blob */
    size_t length,            /* Number of bytes to read */
    char* error_buf,          /* Caller-owned buffer for NUL-terminated error message */
    size_t error_buf_size     /* Size of error_buf in bytes */
);
```

**Return value:** Number of bytes read on success (should equal `length`), or `-1` on error.

The cache provider has full control over how data is served and cached. The cache provider is responsible for serving data and managing its own cache lifecycle. How data is cached, populated, and evicted is entirely up to the cache provider implementation.

The full API contract is defined in [`cpp/azure/azcache_provider/runai_azcache_provider.h`](../cpp/azure/azcache_provider/runai_azcache_provider.h). A test reference implementation is available at [`cpp/azure/azcache_provider/simple_file_cache_test.cc`](../cpp/azure/azcache_provider/simple_file_cache_test.cc).

###### Debugging

Enable debug logging to verify the cache provider is loaded and serving reads:

```bash
export RUNAI_STREAMER_LOG_TO_STDERR=1
export RUNAI_STREAMER_LOG_LEVEL=DEBUG
```

You should see:
```text
AzCacheProvider: auto-discovered cache library: /path/to/site-packages/py_tachyon_client/libStorageDirect.so
AzCacheProvider: cache provider loaded successfully from /path/to/...
```

If the library is not found or fails to load, the streamer falls back to direct Azure Blob Storage access.

#### Streaming from Google cloud storage

##### SDK Authentication

GCS SDK backend is provided through the Python package `runai-model-streamer-gcs`.

To authenticate to GCS, there are multiple configuration options:

1. External Credentials: If you set the `RUNAI_STREAMER_GCS_CREDENTIAL_FILE` environment variable, the
   SDK will load credentials from a JSON file at the path specified (eg: service account credentials)
2. Default Credentials: If you set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable, the google-cloud-cpp
   SDK will read application default credentials from a JSON file at the path specified.
3. Metadata Server: If neither above environment variables are set, the SDK attempts to fetch an auth token from
   the GCP metadata server. This is applicable when running on a GCE, GKE or GAE environment.

See [How Application Default Credentials works](https://cloud.google.com/docs/authentication/application-default-credentials)
for more information.

###### Transport Configuration

By default, the GCS SDK backend communicates using the standard HTTP/JSON transport. To achieve significantly higher throughput and lower latency when running within Google Cloud (e.g., GKE or GCE), you can use [gRPC to interact with Cloud Storage](https://docs.cloud.google.com/storage/docs/enable-grpc-api). gRPC utilizes [direct connectivity](https://docs.cloud.google.com/storage/docs/direct-connectivity) between Compute Engine instances and Cloud Storage buckets, bypassing [Google Front Ends (GFEs)](https://docs.cloud.google.com/docs/security/infrastructure/design#google-frontend-service).

To enable the gRPC client, set the following environment variable (Note: this variable strictly requires a numeric boolean value):
* `RUNAI_STREAMER_GCS_USE_GRPC`: Set this to `1` to enable the gRPC transport, or `0` (the default) to use HTTP/JSON.

**Verifying DirectPath Connectivity:**
If you enable gRPC, we highly recommend verifying that direct connectivity is successfully routing your traffic, as it is needed to achieve optimal performance. If direct connectivity is unavailable in your environment, it is best to leave `RUNAI_STREAMER_GCS_USE_GRPC` unset to fall back to the default transport for the best performance.

You can verify your connection by enabling the gRPC core trace logs with the following environment variables:

```bash
export GRPC_VERBOSITY=DEBUG
export GRPC_TRACE=client_channel,xds
```

In the resulting console output, you are looking for metric blocks related to actual file downloads (e.g. `google.storage.v2.Storage/ReadObject`). To know direct connectivity is set, you need to see the `grpc_target` field in those blocks explicitly targeting `google-c2p:///storage.googleapis.com`. If your actual file downloads are instead targeting standard `dns:///storage.googleapis.com` addresses, traffic is going through the standard public Cloud Load Balancers instead of the unproxied DirectPath route, and you should disable the gRPC client. (Note: You may still see `dns:///storage.googleapis.com` targets for background routing lookups like `RouteLookupService/RouteLookup`, which is perfectly normal as long as the `ReadObject` calls use `google-c2p:///`).

##### HMAC Authentication

S3 compatible HMAC authentication to GCS is provided through the Python package `runai-model-streamer-s3`.

To use HMAC credentials, you can use the S3 backend library, and AWS environment variables.
You should set the following variables:
 * `AWS_ACCESS_KEY_ID`: Set this to the Interoperability Access ID for your GCS bucket
 * `AWS_SECRET_ACCESS_KEY`: Set this to the Interoperability Secret for your GCS bucket
 * `AWS_ENDPOINT_URL`: Set this to `https://storage.googleapis.com`
 * `AWS_EC2_METADATA_DISABLED`: Set this to `true`

See [HMAC keys](https://cloud.google.com/storage/docs/authentication/hmackeys) for more information.

The streamer supports GCS URLs when using HMAC authentication
(eg: `gs://my-bucket/my/file/path.safetensors`).

#### Streaming from S3 compatible storage

To load tensors from S3 compatible object store, define the following environment variables

`RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 AWS_ENDPOINT_URL="your_S3_endpoint" AWS_EC2_METADATA_DISABLED=true`

Setting the environment variable `AWS_ENDPOINT_URL` is mandatory

Setting the environment variable `AWS_EC2_METADATA_DISABLED` is needed in order to avoid a delay of few seconds, which happens only when the aws s3 sdk is used for compatible storage as explained [here](https://github.com/aws/aws-sdk-cpp/issues/1410)   

####  Troubleshooting

For the object storage SDK trace logs pass the environment variable `RUNAI_STREAMER_S3_TRACE=1` - this will create a log file in the location of the application

For the streamer internal logs pass the environment variables `RUNAI_STREAMER_LOG_TO_STDERR=1 RUNAI_STREAMER_LOG_LEVEL=DEBUG`

### CPU Memory Capping

The streamer allocates a buffer on the CPU Memory for storing the tensors before moving them to the GPU Memory. Control the size of the allocated buffer by using the environment variable `RUNAI_STREAMER_MEMORY_LIMIT`.

#### Unlimited CPU Memory

`RUNAI_STREAMER_MEMORY_LIMIT=-1`

The default value. The size of the allocated CPU Memory buffer is equal to the size of the safetensor file (without the file header) and there is no memory reuse between multiple `get_tensors()` requests. Use this option for maximum performance and fastest model streaming times.

#### Min

`RUNAI_STREAMER_MEMORY_LIMIT=0`

The size of allocated CPU memory is minimal and is equal to the size of the largest tensor in the file. The buffer is reused between all `get_tensors()` requests.

#### Limited

`RUNAI_STREAMER_MEMORY_LIMIT=NUMBER`

Use this option to control the size of the buffer by setting the value to a specific memory size. For example, limit the buffer to 4GB by setting `RUNAI_STREAMER_MEMORY_LIMIT=4000000000`. In this case, the streamer reuses the buffer memory between multiple `get_tensors()` requests.

> **Warning:** You cannot limit the buffer to sizes smaller than the size of the largest tensor in the file.
