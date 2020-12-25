#ifndef SPARSECONVNET_SRC_DATACONFIG_H_
#define SPARSECONVNET_SRC_DATACONFIG_H_

#include <boost/any.hpp>
#include <boost/filesystem.hpp>


class DataConfig {
  // Config class for keeping necessary information for data processing and training.
 public:
   DataConfig(const char* data_path, int64_t num_epochs, int64_t batch_size):
       data_path_(data_path), num_epochs_(num_epochs), batch_size_(batch_size) {
    if(num_epochs < 1) throw std::invalid_argument("num_epochs must be > 0");
    if(batch_size < 1) throw std::invalid_argument("batch_size must be > 0");

    // Check if path exists and is of a regular file.
    boost::filesystem::path pathObj(data_path);
    if(!boost::filesystem::exists(pathObj)) throw std::invalid_argument(
        "Cannot find data_path, does it exist?");
    };

    const char* DataPath();
    int64_t BatchSize();
    int64_t NumEpochs();
    int64_t BatchesPerEpoch(int64_t dataset_size);

 private:
   const char* data_path_;
   int64_t num_epochs_;
   int64_t batch_size_;
};
#endif // SPARSECONVNET_SRC_DATACONFIG_H_