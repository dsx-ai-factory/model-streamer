
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <ostream>

#include "common/submission/submission_id.h"
#include "common/responder/responder.h"
#include "common/storage_uri/storage_uri.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "common/response_code/response_code.h"
#include "common/shared_queue/shared_queue.h"
//#include "common/range/range.h"

#include "streamer/impl/config/config/config.h"
#include "streamer/impl/task/task.h"
#include "streamer/impl/reader/reader.h"
#include "streamer/impl/chunk_splitter/chunk_splitter.h"

namespace runai::llm::streamer::impl
{

// Batch represents a single range in the file, and is used by a single worker thread
// The range may contain several sub ranges (requests) which can be whole or partial
// Each sub range (or part of it) is represented by a task
//
//         [                Batch range                     ]
//   [ ... request 1  ][request 2][   request 3   ][ request 4 ....]
//         [task 1    ][  task 2 ][    task 3     ][ task 4 ]

using Tasks = std::vector<Task>;


struct Batch
{
  struct Range : common::Range
  {
      Range() = default;
      Range(size_t start_offset, size_t end_offset);
      Range(const Tasks & tasks);

      size_t end;

   private:
      static size_t calculate_start(const Tasks & tasks);
      static size_t calculate_end(const Tasks & tasks);
  };

  Batch() = default;
  Batch(Batch &&) = default;
  Batch & operator=(Batch &&) = default;

  // chunk_bytesize is the size the BACKEND will request - s3_block_bytesize for object storage, the
  // file system's async chunk size otherwise. It is also where the tasks were cut, so the chunks below
  // cover whole tasks. Batches passes the one value it used for both.
  //
  // NO DEFAULT, deliberately. A default would let a caller construct a Batch without saying which
  // backend it is for, and quietly get the wrong grouping - which is precisely the bug that shipped
  // when Batches cut object-storage tasks at the file system chunk size. Requiring the argument makes
  // every construction state it, so the mistake is a compile error rather than a silent misgrouping
  // that still passes every test.
  Batch(SubmissionId submission_id,
        unsigned workload_index,
        unsigned file_index,
        const std::string & path,
        const common::s3::S3ClientWrapper::Params & params,
        const Tasks && tasks,
        std::shared_ptr<common::Responder> responder,
        std::shared_ptr<const Config> config,
        size_t chunk_bytesize);

  // total number of requested bytes
  size_t total_bytes() const;

  // end offset of the batch
  size_t end_offset() const;

  // read the batch synchronously
  void execute(std::atomic<bool> & stopped);

  // handle response from the reader
  void handle_response(const common::backend_api::Response & response, const Task * task_ptr);

  // handle error
  void handle_error(common::ResponseCode response_code);

  // notify tasks until file offset
  void finished_until(size_t file_offset, common::ResponseCode ret = common::ResponseCode::Success);
  unsigned finished_until() const;

  bool is_object_storage() const;

  // id of the owning submission (one runai_file_streamer_request call); stamped on every response
  SubmissionId submission_id = 0;

  unsigned workload_index;

  // source file
  unsigned file_index;
  std::string path;

  // s3 parameters
  const common::s3::S3ClientWrapper::Params object_storage_params;

  const Tasks tasks;

  // One read request each, covering whole tasks. Built here rather than by each worker: the grouping
  // is decided when the tasks are cut, so re-deriving it per backend is how two implementations of
  // one rule drift apart.
  //
  // Covers only the NON-EMPTY tasks. A zero-sized range owes a response but reads nothing, so its
  // task appears in no chunk and must be completed at enqueue - see chunk_splitter.h.
  const std::vector<Chunk> chunks;

  // range in file
  Range range;

  std::shared_ptr<common::Responder> responder;

  std::shared_ptr<const Config> config;

 private:
  void read(const Config & config, std::atomic<bool> & stopped);

  // handle response from a single task
  void handle_task_response(const common::ResponseCode response_code, const Task * task_ptr);

 private:
  // index of first unfinished task
  unsigned _unfinished = 0;

  std::unique_ptr<Reader> _reader;
};

std::ostream & operator<<(std::ostream & os, const Batch::Range & r);
std::ostream & operator<<(std::ostream & os, const Batch & r);

}; // namespace runai::llm::streamer::impl
