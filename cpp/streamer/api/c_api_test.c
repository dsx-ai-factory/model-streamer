// Compiled by the C compiler, not the C++ one, and linked against the library.
//
// Two things only a test like this can catch:
//
//   The headers are plain C. Every other translation unit we build is C++, so a C++-ism in a shipped
//   header compiles everywhere in this repository and fails only for the consumer. The listing
//   example in streamer.h had exactly that defect - nullptr, a lambda, emplace_back.
//
//   The symbols are exported unmangled. extern "C" used to be gated on _RUNAI_STREAMER_SO, which only
//   the shared object defined, so the static library the C++ tests link carried MANGLED names. Every
//   C++ test passed while no C program could link at all. Only a C caller sees it.

#include <streamer/streamer.h>

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char * what)
{
    if (!ok)
    {
        printf("FAIL: %s\n", what);
        failures += 1;
    }
}

int main(void)
{
    void * streamer = NULL;

    int rc = runai_file_streamer_start(&streamer);
    check(rc == RUNAI_FILE_STREAMER_RESPONSE_SUCCESS, "start returns Success");
    check(streamer != NULL, "start sets the handle");

    const char * message = runai_file_streamer_response_str(RUNAI_FILE_STREAMER_RESPONSE_SUCCESS);
    check(message != NULL && strlen(message) > 0, "response_str returns a message");

    // A submission of no files: accepted, owes no responses, and reaches every parameter of the
    // signature - including the device struct, which crosses BY VALUE and is the part most likely to
    // break silently if the struct's layout ever changes.
    RunaiFileStreamerDevice host;
    host.type = RUNAI_FILE_STREAMER_DEVICE_CPU;
    host.id = 0;

    RunaiFileStreamerSubmissionId id = 12345;
    rc = runai_file_streamer_request(streamer, &id, 0, NULL, NULL, NULL, NULL, NULL, host);
    check(rc == RUNAI_FILE_STREAMER_RESPONSE_SUCCESS, "empty request is accepted");

    runai_file_streamer_end(streamer);

    printf("%s\n", failures == 0 ? "c_api_test OK" : "c_api_test FAILED");
    return failures == 0 ? 0 : 1;
}
