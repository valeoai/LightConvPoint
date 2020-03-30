# pragma once
#include <torch/extension.h>

torch::Tensor knn(const torch::Tensor points, const torch::Tensor queries, const size_t K);

std::vector<torch::Tensor> random_pick_knn(const torch::Tensor points, const size_t nqueries,
		const size_t K);

std::vector<torch::Tensor> convpoint_pick_knn(const torch::Tensor points, const size_t nqueries,
		const size_t K);
std::vector<torch::Tensor> convpoint_pick_knn_dev(const torch::Tensor points, const size_t nqueries,
		const size_t K);


std::vector<torch::Tensor> farthest_pick_knn(const torch::Tensor points, const size_t nqueries,
		const size_t K);

std::vector<torch::Tensor> quantized_pick_knn(const torch::Tensor points, const size_t nqueries,
		const size_t K);