#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "common/backend_api/object_storage/object_storage.h"

namespace runai::llm::streamer::common::s3
{

// Credentials - a general key -> value dictionary of object-storage client-configuration parameters
// (credentials and related config). Keys are the plugin's canonical config-parameter names; arbitrary keys
// are carried through to the backend unchanged, so a new parameter needs no change here - only the plugin
// must handle it (and validate it; see obj_supported_config_params). A typed convenience API for the
// currently-supported S3 credentials is layered on top (the 5-argument constructor + named key constants +
// endpoint()), so existing typed construction sites are unchanged.
struct Credentials
{
    // -- General dictionary API --

    Credentials() = default;

    // Build from parallel key/value arrays (the C API form). Any entry with a null key or null value is
    // skipped (logged).
    Credentials(const char ** keys, const char ** values, unsigned num_params);

    void set(const std::string & key, const std::string & value);
    bool has(const std::string & key) const;
    std::optional<std::string> get(const std::string & key) const;
    const std::map<std::string, std::string> & params() const;

    // -- Specific credentials API (currently-supported S3 credentials) --
    // Convenience over the dictionary: each non-null argument is stored under its canonical key.
    Credentials(const char * access_key_id, const char * secret_access_key, const char * session_token, const char * region, const char * endpoint);

    // endpoint is a dedicated field of the backend client config (endpoint_url), not an initial param.
    std::optional<std::string> endpoint() const;

    // Push every parameter EXCEPT endpoint as an initial client-config param (endpoint -> endpoint_url).
    void to_object_client_config(std::vector<common::backend_api::ObjectConfigParam_t> & config) const;

    // Value equality over the whole dictionary. Used by the streamer to detect a second
    // runai_file_streamer_set_credentials call that changes the (already-applied) credentials.
    bool operator==(const Credentials & other) const;
    bool operator!=(const Credentials & other) const;

    // canonical config-parameter keys for the supported credentials
    static const char * const ACCESS_KEY_ID_KEY;
    static const char * const SECRET_ACCESS_KEY_KEY;
    static const char * const SESSION_TOKEN_KEY;
    static const char * const REGION_KEY;
    static const char * const ENDPOINT_KEY;

 private:
    std::map<std::string, std::string> _params;
};

}; //namespace runai::llm::streamer::common::s3
