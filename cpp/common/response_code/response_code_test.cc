#include "common/response_code/response_code.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <map>
#include <string>
#include <array>

#include "utils/logging/logging.h"
#include "utils/random/random.h"

namespace runai::llm::streamer::common
{

TEST(Response, Description)
{
    constexpr std::array<const char *, static_cast<size_t>(ResponseCode::__Max)> __strings = {
        "Request sent successfuly",
        "Finished all responses",
        "File access error",
        "End of file reached",
        "S3 not supported",
        "GLIBC version should be at least 2.28",
        "Increase process fd limit or decrease the concurrency level. Recommended value for the streamer alone is the concurrency multiplied by 64, in addition to your application fd usage",
        "Invalid request parameters",
        "Empty request parameters",
        "Streamer is handling previous request",
        "CA bundle file not found",
        "Unknown Error",
        "Error loading object storage plugin",
        "GCS not supported"
    };

    // errors

    for (auto response_code : {ResponseCode::FileAccessError, ResponseCode::EofError, ResponseCode::InvalidParameterError, ResponseCode::EmptyRequestError, ResponseCode::BusyError, ResponseCode::UnknownError, ResponseCode::FinishedError, ResponseCode::S3NotSupported, ResponseCode::GlibcPrerequisite, ResponseCode::InsufficientFdLimit, ResponseCode::CaFileNotFound, ResponseCode::ObjPluginLoadError, ResponseCode::GCSNotSupported} )
    {
        std::string str = description(static_cast<int>(response_code));

        const auto expected = __strings[static_cast<int>(response_code)];

        EXPECT_EQ(str, expected);
    }
}

TEST(Description, Invalid)
{
    for (int response_code : { utils::random::number<int>(1000, 10000), utils::random::number<int>(-1000, -1) } )
    {
        std::string str = description(static_cast<int>(response_code));
        EXPECT_EQ(str, "Invalid response code");
    }
}

// Every code must have a description, and the array is indexed by the enum - so a code appended
// without a message would read whatever follows the array, or shift every message after it.
//
// The array's SIZE is checked by the compiler (it is sized from __Max), but its ORDER is not. This
// walks the whole enum and asserts each answer is a real message rather than the "invalid" fallback,
// which is what a missing or misplaced entry produces.
TEST(Description, Every_Code_Has_Its_Own_Message)
{
    std::set<std::string> seen;

    for (int code = 0; code < static_cast<int>(ResponseCode::__Max); ++code)
    {
        const std::string message = description(code);

        EXPECT_NE(message, "Invalid response code") << "code " << code << " has no message";
        EXPECT_FALSE(message.empty()) << "code " << code << " has an empty message";
        EXPECT_TRUE(seen.insert(message).second)
            << "code " << code << " repeats the message of an earlier one, which is what a missing"
            << " entry looks like - every code after it shifts by one";
    }
}

// The code an engine reports when its ring or context fails for good. Named here because the point of
// it is that it is NOT UnknownError: that one tells a caller to abort everything and treat it as our
// bug, and a dead engine on one mount is neither.
TEST(Description, FsAsyncEngineError)
{
    EXPECT_NE(ResponseCode::FsAsyncEngineError, ResponseCode::UnknownError);
    EXPECT_NE(ResponseCode::FsAsyncEngineError, ResponseCode::FileAccessError);

    const std::string message = description(static_cast<int>(ResponseCode::FsAsyncEngineError));
    EXPECT_NE(message, "Invalid response code");
    EXPECT_NE(message.find("synchronous reader"), std::string::npos)
        << "the message must say what happens next, since the caller can simply ask again";
}

TEST(Description, InternalRetryableFileAccessError)
{
    EXPECT_STREQ(description(static_cast<int>(ResponseCode::RetryableFileAccessError)),
                 "Retryable object storage file access error");
}

}; // namespace runai::llm::streamer::common
