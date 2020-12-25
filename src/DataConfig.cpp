#include "DataConfig.h"


int64_t DataConfig::BatchSize() {
  return batch_size_;
}

int64_t DataConfig::NumEpochs() {
  return num_epochs_;
}

int64_t DataConfig::BatchesPerEpoch(int64_t dataset_size) {
  return  std::ceil(dataset_size / static_cast<double>(batch_size_));
}

const char* DataConfig::DataPath() {
  return data_path_;
}
