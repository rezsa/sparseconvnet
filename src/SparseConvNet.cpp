#include "SparseConvNet.h"


SparseConvNet::SparseConvNet(float sparsity) {
  if(sparsity < 0 && sparsity >= 1) throw std::invalid_argument(
      "sparsity must be in range [0, 1)");
  sparsity_ = sparsity;

  // Must register module to be able to access the model trainable params attribute.
  conv1_ = register_module("conv1",
      nn::Conv2d(nn::Conv2dOptions(1, 16, 3).stride(1).padding(1).bias(false)));
  bn1_ = register_module("bn1", nn::BatchNorm2d(conv1_dim_[0]));

  conv2_ = register_module("conv2",
      nn::Conv2d(nn::Conv2dOptions(16, 16, 3).stride(1).padding(1).bias(false)));
  bn2_ = register_module("bn2", nn::BatchNorm2d(conv2_dim_[0]));

  conv3_ = register_module("conv3",
      nn::Conv2d(nn::Conv2dOptions(16, 32, 3).stride(1).padding(1).bias(false)));
  bn3_ = register_module("bn3", nn::BatchNorm2d(conv3_dim_[0]));

  conv4_ = register_module("conv4",
      nn::Conv2d(nn::Conv2dOptions(32, 32, 3).stride(1).padding(1).bias(false)));
  bn4_ = register_module("bn4", nn::BatchNorm2d(conv4_dim_[0]));

  conv5_ = register_module("conv5",
      nn::Conv2d(nn::Conv2dOptions(32, 64, 3).stride(1).padding(1).bias(false)));
  bn5_ = register_module("bn5", nn::BatchNorm2d(conv5_dim_[0]));

  conv6_ = register_module("conv6",
      nn::Conv2d(nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)));
  bn6_ = register_module("bn6", nn::BatchNorm2d(conv6_dim_[0]));

  conv7_ = register_module("conv7",
      nn::Conv2d(nn::Conv2dOptions(64, 128, 3).stride(1).padding(1).bias(false)));
  bn7_ = register_module("bn7", nn::BatchNorm2d(conv7_dim_[0]));

  conv8_ = register_module("conv8",
      nn::Conv2d(nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)));
  bn8_ = register_module("bn8", nn::BatchNorm2d(conv8_dim_[0]));

  flatten_ = register_module("flatten", nn::Flatten());
  dense1_ = register_module("dense1", nn::Linear(conv8_dim_[0], dense1_dim_[0]));
  dense2_ = register_module("dense2", nn::Linear(dense1_dim_[0], dense2_dim_[0]));

  logits_ = register_module("logits", nn::Linear(dense1_dim_[0], 10));

  conv1_mask_ = MakeMask(conv1_dim_);
  conv3_mask_ = MakeMask(conv3_dim_);
  conv5_mask_ = MakeMask(conv5_dim_);
  conv7_mask_ = MakeMask(conv7_dim_);

  dense1_mask_ = MakeMask(dense1_dim_);
  dense2_mask_ = MakeMask(dense2_dim_);
};

Tensor SparseConvNet::MakeMask(vector<int64_t> &shape) {
  // Function for creating a sparse binary tensor mask with the given shape.
  // The sparsity of the mask is based on the sparsity arg at initialization.
  int64_t size = 1;
  for(int64_t i : shape){
    size *= i;
  }
  Tensor sparser = torch::ones(size);

  // Creates a list of n random indices without replacement where n is
  // the size of the shape provided.
  Tensor random_indices = at::randperm(size).to(kLong);

  // Takes the first k of the random indices based on the sparsity factor.
  int64_t num_sparse_indices = (int64_t)(sparsity_ * size + 0.5f);

  // Creates the binary tensor mask based on the random sparse indices.
  sparser.index_put_(
      {random_indices.index({Slice(None, num_sparse_indices)})}, 0.);
  return sparser.reshape(shape);
}

Tensor SparseConvNet::forward(Tensor x) {
  // Conv block 1.
  x = relu(bn1_(conv1_(x)));
  x *= conv1_mask_;
  x = relu(bn2_(conv2_(x)));
  x = nn::MaxPool2d(nn::MaxPool2dOptions(2).stride(2))(x);

  // Conv block 2.
  x = relu(bn3_(conv3_(x)));
  x *= conv3_mask_;
  x = relu(bn4_(conv4_(x)));
  x = nn::MaxPool2d(nn::MaxPool2dOptions(2).stride(2).padding(1))(x);

  // Conv block 3.
  x = relu(bn5_(conv5_(x)));
  x *= conv5_mask_;
  x = relu(bn6_(conv6_(x)));
  x = nn::MaxPool2d(nn::MaxPool2dOptions(2).stride(2))(x);

  // Conv block 4.
  x = relu(bn7_(conv7_(x)));
  x *= conv7_mask_;
  x = relu(bn8_(conv8_(x)));
  x = nn::MaxPool2d(nn::MaxPool2dOptions(4).stride(4))(x);

  // Dense section.
  x = dense1_(flatten_(x));
  x *= dense1_mask_;
  x = dense2_(x);
  x *= dense2_mask_;

  return logits_(x);
 }
