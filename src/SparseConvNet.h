#ifndef SPARSECONVNET_SRC_SPARSECONVNET_H_
#define SPARSECONVNET_SRC_SPARSECONVNET_H_

#include <torch/torch.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;


class SparseConvNet : public nn::Module {
  // Class for creating a Sparse convolutional network (SparseConvNet).
  // The model architecture is a VGG-style convnet that has an eigen mask
  // in between layers to induce sparsity in the network.
 public:
   // The SparseConvNet intitializer takes a sparisty factor that must be in
   // the range [0, 1), i.e. 0 for no sparsity and less than 1 (completely sparse).
   SparseConvNet(float sparsity);
   Tensor forward(Tensor x);

 private:
   // Dimensions: {Channel, Width, Height}.
   vector<int64_t> conv1_dim_{16, 28, 28};
   vector<int64_t> conv2_dim_{16, 28, 28};
   vector<int64_t> conv3_dim_{32, 14, 14};
   vector<int64_t> conv4_dim_{32, 14, 14};
   vector<int64_t> conv5_dim_{64, 8, 8};
   vector<int64_t> conv6_dim_{64, 8, 8};
   vector<int64_t> conv7_dim_{128, 4, 4};
   vector<int64_t> conv8_dim_{128, 4, 4};

   vector<int64_t> dense1_dim_{128};
   vector<int64_t> dense2_dim_{128};

   float sparsity_;

   Tensor conv1_mask_;
   Tensor conv2_mask_;
   Tensor conv3_mask_;
   Tensor conv4_mask_;
   Tensor conv5_mask_;
   Tensor conv6_mask_;
   Tensor conv7_mask_;
   Tensor conv8_mask_;

   Tensor dense1_mask_;
   Tensor dense2_mask_;


   nn::Flatten flatten_{nullptr};
   nn::Linear dense1_{nullptr}, dense2_{nullptr}, logits_{nullptr};

   nn::Conv2d conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr}, conv4_{nullptr};
   nn::Conv2d conv5_{nullptr}, conv6_{nullptr}, conv7_{nullptr}, conv8_{nullptr};
   nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr}, bn3_{nullptr}, bn4_{nullptr};
   nn::BatchNorm2d bn5_{nullptr}, bn6_{nullptr}, bn7_{nullptr}, bn8_{nullptr};

   Tensor MakeMask(vector<int64_t> &shape);

  // For NLLLoss, logits must be passed through log_softmax function below.
  //  nn::LogSoftmax log_softmax;
};
#endif // SPARSECONVNET_SRC_SPARSECONVNET_H_