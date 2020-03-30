#include "knn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn", &knn, "knn computation");
  m.def("random_pick_knn", &random_pick_knn, "knn computation with random picking");
  m.def("convpoint_pick_knn", &convpoint_pick_knn, "knn computation as in ConvPoint");
  m.def("convpoint_pick_knn_dev", &convpoint_pick_knn_dev, "knn computation as in ConvPoint");
  m.def("farthest_pick_knn", &farthest_pick_knn, "knn computation as in PointNet++");
  m.def("quantized_pick_knn", &quantized_pick_knn, "knn computation with voxels");
}