# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Dict, Optional

import os
import boto3

AWS_CA_BUNDLE_ENV = "AWS_CA_BUNDLE"
RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR = "RUNAI_STREAMER_S3_UNSIGNED"
RUNAI_STREAMER_NO_BOTO3_SESSION_ENV_VAR = "RUNAI_STREAMER_NO_BOTO3_SESSION"

class S3Credentials:
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.region_name = region_name
        self.endpoint = endpoint

def get_credentials(credentials: Optional[S3Credentials] = None) -> Tuple[Optional[boto3.Session], S3Credentials]:
    """
    Resolves S3 credentials, optionally via a boto3 session.

    By default (RUNAI_STREAMER_NO_BOTO3_SESSION=1) no boto3 session is created:
    the credentials are returned as provided and the C++ layer authenticates using
    the AWS default credential provider chain (environment, profile, SSO, IMDS, etc.),
    or the explicitly provided credentials. Set RUNAI_STREAMER_NO_BOTO3_SESSION=0 to
    resolve credentials through a boto3 session in Python and pass the resolved
    (frozen) credentials to the C++ layer.

    Returns:
        - boto3.Session object (or None when no session is created)
        - S3Credentials object with the resolved credentials (or the provided ones)
    """

    # CA bundle resolution (AWS_CA_BUNDLE env or the profile "ca_bundle" setting)
    # is handled by the C++ layer, so no boto3 session is needed for unsigned or
    # no-session modes.
    if os.getenv(RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR, "0") == "1":
        return None, credentials if credentials else S3Credentials()

    if os.getenv(RUNAI_STREAMER_NO_BOTO3_SESSION_ENV_VAR, "1") == "1":
        return None, credentials if credentials else S3Credentials()

    session = boto3.Session(
        aws_access_key_id=credentials.access_key_id if credentials else None,
        aws_secret_access_key=credentials.secret_access_key if credentials else None,
        aws_session_token=credentials.session_token if credentials else None,
        region_name=credentials.region_name if credentials else None
    )

    # Retrieve the actual credentials (could be from env vars, IAM role, etc.)
    resolved_credentials = session.get_credentials()
    frozen_creds = resolved_credentials.get_frozen_credentials() if resolved_credentials else None

    # Create a new S3Credentials object with the resolved credentials
    new_credentials = S3Credentials(
        access_key_id=frozen_creds.access_key if frozen_creds else None,
        secret_access_key=frozen_creds.secret_key if frozen_creds else None,
        session_token=frozen_creds.token if frozen_creds else None,
        region_name=session.region_name,
        endpoint=credentials.endpoint if credentials else None,
    )

    return session, new_credentials