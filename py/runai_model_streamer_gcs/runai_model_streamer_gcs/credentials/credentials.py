# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Dict, Optional
from enum import Enum, auto

import google.auth
import google.auth.credentials

import os

GCS_CREDENTIAL_TYPE = "GCS_CREDENTIAL_TYPE"
GCS_SA_KEY_PATH = "GCS_SA_KEY_PATH"
# TODO: Pass credentials through to the C++ API library.
RUNAI_STREAMER_GCS_CREDENTIAL_FILE = "RUNAI_STREAMER_GCS_CREDENTIAL_FILE"
RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS = "RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS"

class CredentialType(Enum):
    # Credentials provided explicitly via a JSON file.
    SERVICE_ACCOUNT_JSON = auto()
    # Workload Identity or Application Default Credentials.
    DEFAULT_CREDENTIALS = auto()
    # Workload Identity or Application Default Credentials.
    ANONYMOUS_CREDENTIALS = auto()

class GCSCredentials:
    def __init__(
        self,
        credential_type: CredentialType,
        sa_key_file: Optional[str] = None,
    ):
        self.credential_type = credential_type
        if sa_key_file:
            assert credential_type == CredentialType.SERVICE_ACCOUNT_JSON
            self.sa_key_path = sa_key_file

    def serialized_credentials(self) -> Dict[str, str]:
        return {
            GCS_CREDENTIAL_TYPE: self.credential_type.value,
            GCS_SA_KEY_PATH: self.sa_key_file,
        }

    def gcp_credentials(self) -> google.auth.credentials.Credentials:
        credentials = None
        if self.credential_type == CredentialType.ANONYMOUS_CREDENTIALS:
            credentials = google.auth.credentials.AnonymousCredentials()
        elif self.credential_type == CredentialType.SERVICE_ACCOUNT_JSON:
            credentials, _ = google.auth.load_credentials_from_file(self.sa_key_path)
        else:
            credentials, _ = google.auth.default()

        return credentials

def getenv_as_bool(key: str) -> bool:
    return (os.getenv(key, "False").lower() in ("true", "1"))

def get_credentials() -> GCSCredentials:
    """
    Creates a GCS ServiceAccount credential if the environment variable RUNAI_STREAMER_GCS_CREDENTIAL_FILE is set.
    If the variable is not set, returns None and the original credentials.

    Returns:
        - GCSCredentials object with the resolved credentials (or original if session not created)
    """

    use_anon_creds = getenv_as_bool(RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS)
    if use_anon_creds:
        return GCSCredentials(CredentialType.ANONYMOUS_CREDENTIALS)

    sa_key_file = os.getenv(RUNAI_STREAMER_GCS_CREDENTIAL_FILE, default=None)
    if sa_key_file:
        return GCSCredentials(CredentialType.SERVICE_ACCOUNT_JSON, sa_key_file)

    return GCSCredentials(CredentialType.DEFAULT_CREDENTIALS)
