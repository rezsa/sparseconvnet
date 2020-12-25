#include "DataConfig.h"
#include "SparseConvNet.h"

#include <iostream>

#include <torch/torch.h>

using namespace torch;


int main(int argc, const char* argv[]) {

  // Interval to log loss results.
  const int64_t kLogInterval = 10;

  // Batch size for test data for computing test metrics.
  // OK for this to be large since we're not worried about optimizer steps.
  const int64_t kTestBatchSize = 1000;

  // Setting defaults if no command line args.
  // Command line args:
  // 1: data directory
  // 2: num epochs
  // 3: batch size
  // 4: image pixel value normalization mean
  // 5: image pixel value normalization stdev
  // 6: level of SparseConvNet sparsity and must be in range [0, 1)
  // 7: SGD learning rate
  // 8: num workers for loading data
  int const default_num_args = 9;
  const char* default_args[] = {"./main", "../data", "5", "32", "0.5",
      "0.5", "0.2", "0.001", "2"};
  if (argc == 1){
    argc = default_num_args;
    argv = default_args;
  }

  // Select CPU device.
  torch::Device device(torch::kCPU);

  // Get data config args from cmd and configure.
  DataConfig data_gen = DataConfig(argv[1], std::atoi(argv[2]),
      std::atoi(argv[3]));
  int64_t batch_size = data_gen.BatchSize();
  int64_t num_epochs = data_gen.NumEpochs();

  // Create data generator.
  float normalize_mean = std::atof(argv[4]);
  float normalize_stdev = std::atof(argv[5]);

  // Training set.
  auto dataset_train = torch::data::datasets::MNIST(
      data_gen.DataPath(), data::datasets::MNIST::Mode::kTrain)
      .map(data::transforms::Normalize<>(normalize_mean,normalize_stdev))
      .map(data::transforms::Stack<>());

  int num_workers = std::atoi(argv[8]);
  int64_t batches_per_epoch = data_gen.BatchesPerEpoch(
      dataset_train.size().value());
  auto train_batch_generator = data::make_data_loader(
      std::move(dataset_train),
      data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));

  // Build model.
  float sparsity = std::atof(argv[6]);
  auto model = std::make_shared<SparseConvNet>(sparsity);

  // Setup SGD optimizer.
  float lr = std::atof(argv[7]);
  optim::SGD optimizer(model->parameters(), optim::SGDOptions(lr));

  // Setup loss function for multiclass classification.
  nn::CrossEntropyLoss xentropy_loss = nn::CrossEntropyLoss();
  // If NLLLoss is to be used, logits must be passed through log_softmax.
  // torch::nn::NLLLoss nll_loss = torch::nn::NLLLoss();

  // Training loop.
  for (int64_t epoch = 1; epoch <= num_epochs; ++epoch) {
    int64_t batch_index = 0;
    for (data::Example<>& batch : *train_batch_generator) {
      // Reset gradients.
      optimizer.zero_grad();

      // Scuttle data to device.
      Tensor images_train = batch.data.to(device);
      Tensor labels_train = batch.target.to(device);

      // Call forward pass.
      Tensor output_train = model->forward(images_train);
      Tensor loss = xentropy_loss(output_train, labels_train);

      // Backpropagation and update weights.
      loss.backward();
      optimizer.step();
      batch_index++;

      // Logging.
      if (batch_index % kLogInterval == 0) {
        std::printf(
            "\r[Epoch: %2ld/%2ld][Step: %3ld/%3ld] Loss: %.4f\n",
            epoch,
            num_epochs,
            batch_index * batch_size,
            batches_per_epoch * batch_size,
            loss.item<float>());
      }
    }
  }
  std::cout << "Finished training." << std::endl;

  // Test set.
  auto dataset_test = data::datasets::MNIST(
      data_gen.DataPath(), data::datasets::MNIST::Mode::kTest)
      .map(data::transforms::Normalize<>(normalize_mean,normalize_stdev))
      .map(data::transforms::Stack<>());

  auto test_batch_generator = data::make_data_loader(
      std::move(dataset_test),
      data::DataLoaderOptions()
          .batch_size(kTestBatchSize).workers(num_workers));

  // Softmax for computing class probabilities from model logits.
  nn::Softmax softmax{1}; 

  int all_correct_preds{0};
  for (data::Example<>& batch : *test_batch_generator) {
      // Scuttle data to device.
      Tensor images_test = batch.data.to(device);
      Tensor labels_test = batch.target.to(device);

      // Call forward pass.
      Tensor output_test = model->forward(images_test);
      // We first call softmax on model output logits to compute class probabilities.
      // We then take argmax of class probs to find target predictions.
      // We then check how many of targets were correct and assign to correct_preds.
      Tensor correct_preds = at::argmax(softmax(output_test), 1) == labels_test;
      all_correct_preds += correct_preds.sum().item<float>();
  }
  // 10000 is the size of the MNIST test dataset.
  std::cout << "Test accuracy (%): " << 100 * all_correct_preds / 10000. << std::endl;
}
