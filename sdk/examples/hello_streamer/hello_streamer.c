#include <streamer/streamer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read the supplied path/URI and verify its bytes against the second argument.
 * Keeping the expected bytes separate also lets CI use this example for S3. */
int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s PATH EXPECTED_CONTENT\n", argv[0]);
        return 1;
    }
    void *streamer = NULL;
    const size_t size = strlen(argv[2]);
    void *buffer = malloc(size ? size : 1);
    if (!buffer) return 1;
    int rc = runai_file_streamer_start(&streamer);
    if (rc != RUNAI_FILE_STREAMER_RESPONSE_SUCCESS) goto done;
    rc = runai_file_streamer_set_fs_strategy(streamer, "sync_buffered");
    if (rc != RUNAI_FILE_STREAMER_RESPONSE_SUCCESS) goto done;
    {
        const char *paths[] = {argv[1]};
        unsigned ranges[] = {1};
        size_t offsets[] = {0};
        size_t sizes[] = {size};
        void *destinations[] = {buffer};
        RunaiFileStreamerDevice device = {RUNAI_FILE_STREAMER_DEVICE_CPU, 0};
        RunaiFileStreamerSubmissionId submitted = 0, received = 0;
        unsigned file = 0, range = 0;
        int finished = 0;
        rc = runai_file_streamer_request(streamer, &submitted, 1, paths, ranges,
                                         offsets, sizes, destinations, device);
        if (rc != RUNAI_FILE_STREAMER_RESPONSE_SUCCESS) goto done;
        rc = runai_file_streamer_response(streamer, &received, &file, &range, &finished, 10000);
        if (rc != RUNAI_FILE_STREAMER_RESPONSE_SUCCESS) goto done;
        if (submitted != received || file != 0 || range != 0 || !finished ||
            memcmp(buffer, argv[2], size) != 0) {
            fprintf(stderr, "Unexpected response or file contents\n");
            rc = RUNAI_FILE_STREAMER_RESPONSE_UNKNOWN_ERROR;
        }
    }
done:
    if (rc != RUNAI_FILE_STREAMER_RESPONSE_SUCCESS)
        fprintf(stderr, "%s\n", runai_file_streamer_response_str(rc));
    if (streamer) runai_file_streamer_end(streamer);
    free(buffer);
    return rc == RUNAI_FILE_STREAMER_RESPONSE_SUCCESS ? 0 : 1;
}
