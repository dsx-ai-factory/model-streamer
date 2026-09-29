# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch, MagicMock

from runai_model_streamer_s3.credentials.credentials import (
    get_credentials,
    AWS_CA_BUNDLE_ENV,
    RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR,
    RUNAI_STREAMER_NO_BOTO3_SESSION_ENV_VAR,
)


def _env_without_unsigned():
    env = os.environ.copy()
    env.pop(RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR, None)
    env.pop(AWS_CA_BUNDLE_ENV, None)
    env.pop(RUNAI_STREAMER_NO_BOTO3_SESSION_ENV_VAR, None)
    return env


class TestGetCredentialsUnsigned(unittest.TestCase):
    @patch("runai_model_streamer_s3.credentials.credentials.boto3")
    def test_unsigned_returns_no_session(self, mock_boto3):
        mock_boto3.Session.return_value._session.get_config_variable.return_value = None
        with patch.dict(os.environ, {RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR: "1"}, clear=False):
            session, _ = get_credentials(None)
        self.assertIsNone(session)

    @patch("runai_model_streamer_s3.credentials.credentials.boto3")
    def test_unsigned_creates_no_session(self, mock_boto3):
        # CA bundle resolution now lives in the C++ layer; unsigned mode simply
        # creates no boto3 session (and no longer touches AWS_CA_BUNDLE)
        env = _env_without_unsigned()
        env[RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR] = "1"
        with patch.dict(os.environ, env, clear=True):
            session, _ = get_credentials(None)
        self.assertIsNone(session)
        mock_boto3.Session.assert_not_called()

    @patch("runai_model_streamer_s3.credentials.credentials.boto3")
    def test_no_boto3_session_disabled_resolves_credentials(self, mock_boto3):
        # RUNAI_STREAMER_NO_BOTO3_SESSION=0 opts back into boto3 resolution
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        mock_session._session.get_config_variable.return_value = None
        mock_boto3.Session.return_value = mock_session
        env = _env_without_unsigned()
        env[RUNAI_STREAMER_NO_BOTO3_SESSION_ENV_VAR] = "0"
        with patch.dict(os.environ, env, clear=True):
            session, _ = get_credentials(None)
        mock_session.get_credentials.assert_called_once()
        self.assertIsNotNone(session)

    @patch("runai_model_streamer_s3.credentials.credentials.boto3")
    def test_no_boto3_session_disabled_returns_resolved_values(self, mock_boto3):
        # RUNAI_STREAMER_NO_BOTO3_SESSION=0: the boto3 session's frozen credentials (from env/profile/IMDS/
        # AssumeRole) are extracted into the returned S3Credentials, alongside region and the provided endpoint
        from runai_model_streamer_s3.credentials.credentials import S3Credentials
        frozen = MagicMock()
        frozen.access_key = "RESOLVED_AKID"
        frozen.secret_key = "RESOLVED_SECRET"
        frozen.token = "RESOLVED_TOKEN"
        resolved = MagicMock()
        resolved.get_frozen_credentials.return_value = frozen
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = resolved
        mock_session.region_name = "us-west-2"
        mock_boto3.Session.return_value = mock_session

        env = _env_without_unsigned()
        env[RUNAI_STREAMER_NO_BOTO3_SESSION_ENV_VAR] = "0"
        provided = S3Credentials(endpoint="https://s3.example.com")
        with patch.dict(os.environ, env, clear=True):
            session, creds = get_credentials(provided)

        self.assertIsNotNone(session)
        self.assertEqual(creds.access_key_id, "RESOLVED_AKID")
        self.assertEqual(creds.secret_access_key, "RESOLVED_SECRET")
        self.assertEqual(creds.session_token, "RESOLVED_TOKEN")
        self.assertEqual(creds.region_name, "us-west-2")
        self.assertEqual(creds.endpoint, "https://s3.example.com")

    @patch("runai_model_streamer_s3.credentials.credentials.boto3")
    def test_no_boto3_session_default_returns_no_session(self, mock_boto3):
        # Default: no boto3 session is created; the C++ layer self-authenticates
        with patch.dict(os.environ, _env_without_unsigned(), clear=True):
            session, creds = get_credentials(None)
        self.assertIsNone(session)
        mock_boto3.Session.assert_not_called()
        self.assertIsNotNone(creds)

    @patch("runai_model_streamer_s3.credentials.credentials.boto3")
    def test_no_boto3_session_enabled_returns_no_session(self, mock_boto3):
        env = _env_without_unsigned()
        env[RUNAI_STREAMER_NO_BOTO3_SESSION_ENV_VAR] = "1"
        with patch.dict(os.environ, env, clear=True):
            session, _ = get_credentials(None)
        self.assertIsNone(session)
        mock_boto3.Session.assert_not_called()

    @patch("runai_model_streamer_s3.credentials.credentials.boto3")
    def test_default_returns_provided_credentials_unchanged(self, mock_boto3):
        from runai_model_streamer_s3.credentials.credentials import S3Credentials
        provided = S3Credentials(access_key_id="AKID", secret_access_key="SECRET")
        with patch.dict(os.environ, _env_without_unsigned(), clear=True):
            session, creds = get_credentials(provided)
        self.assertIsNone(session)
        self.assertIs(creds, provided)
        mock_boto3.Session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
