/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <streamer/streamer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A small, bounded read makes this example independent of file size and alignment.
 * Compare with stdio so success proves that the library delivered the expected bytes.
 */
int main(int argc, char ** argv)
{
    unsigned char expected[4096];
    unsigned char actual[4096];
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <local-file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    FILE * file = fopen(argv[1], "rb");
    if (file == NULL)
    {
        perror("open input");
        return EXIT_FAILURE;
    }
    size_t size = fread(expected, 1, sizeof(expected), file);
    int read_failed = ferror(file);
    int close_failed = fclose(file);
    if (read_failed || close_failed)
    {
        fprintf(stderr, "Failed to read reference bytes\n");
        return EXIT_FAILURE;
    }

    void * streamer = NULL;
    int rc = runai_file_streamer_start(&streamer);
    if (rc != RUNAI_FILE_STREAMER_RESPONSE_SUCCESS)
    {
        fprintf(stderr, "start: %s\n", runai_file_streamer_response_str(rc));
        return EXIT_FAILURE;
    }
    /* Buffered I/O accepts stack buffers, arbitrary sizes, and hosts without io_uring. */
    rc = runai_file_streamer_set_fs_strategy(streamer, "sync_buffered");
    RunaiFileStreamerSubmissionId submitted = 0;
    const char * paths[] = {argv[1]};
    unsigned ranges[] = {1};
    size_t offsets[] = {0};
    size_t sizes[] = {size};
    void * destinations[] = {actual};
    RunaiFileStreamerDevice device = {RUNAI_FILE_STREAMER_DEVICE_CPU, 0};
    if (rc == RUNAI_FILE_STREAMER_RESPONSE_SUCCESS)
    {
        rc = runai_file_streamer_request(streamer, &submitted, 1, paths, ranges,
                                        offsets, sizes, destinations, device);
    }
    int valid_response = 0;
    if (rc == RUNAI_FILE_STREAMER_RESPONSE_SUCCESS)
    {
        RunaiFileStreamerSubmissionId answered = 0;
        unsigned file_index = 0;
        unsigned range_index = 0;
        int done = 0;
        rc = runai_file_streamer_response(streamer, &answered, &file_index,
                                         &range_index, &done, 30000);
        valid_response = answered == submitted && file_index == 0 && range_index == 0 && done;
    }
    /* Destinations must remain valid until end returns, including on errors/timeouts. */
    runai_file_streamer_end(streamer);
    if (rc != RUNAI_FILE_STREAMER_RESPONSE_SUCCESS)
    {
        fprintf(stderr, "read: %s\n", runai_file_streamer_response_str(rc));
        return EXIT_FAILURE;
    }
    if (!valid_response || memcmp(actual, expected, size) != 0)
    {
        fprintf(stderr, "Unexpected response or file contents\n");
        return EXIT_FAILURE;
    }
    printf("Verified %zu bytes from %s\n", size, argv[1]);
    return EXIT_SUCCESS;
}
