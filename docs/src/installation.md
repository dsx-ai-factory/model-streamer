## Installation

Model Streamer is a Python package available for installation in PyPi. Installing the streamer is done in your Python project using `pip` or by adding `runai-model-streamer` to your `requirements.txt` file.

### Setup

The Run:ai Streamer comes with precompiled C++ code compatible to run on any Linux machine. Installing the streamer with the pip command is supported for x86_64 processors.

### Installation

To install the streamer, run:

```bash
pip install runai-model-streamer
```

> **Note:** To install a specific version of the package, use `runai-model-streamer==0.3.1`. Visit our PyPi or GitHub repository for the list of available versions.

To stream models from object storage, run the following command as well:

```bash
pip install 'runai-model-streamer[s3]'
```

> **Warning:** Make sure you install the S3 dependency in the same version of your runai-model-streamer by running `pip install 'runai-model-streamer[s3]==0.3.1'`.

<a id="azureCapabilityInstallation"></a>

### Azure Blob Storage

To stream models from Azure Blob Storage, install the Azure package:

```bash
pip install 'runai-model-streamer[azure]'
```

> **Warning:** Make sure you install the Azure dependency in the same version of your runai-model-streamer by running `pip install 'runai-model-streamer[azure]==0.3.1'`. 

### Matching versions

The object storage packages ship a shared library that `runai-model-streamer` loads at runtime. They
are one product built together, not independent plugins, so **always install them at the same version
as `runai-model-streamer`**:

```bash
pip install runai-model-streamer==0.17.0 'runai-model-streamer[s3]==0.17.0'
```

Mixed versions are not supported. `pip` does not prevent them, and the streamer cannot detect every
mismatch, so a mixed installation may fail to load the backend or configure it incorrectly.