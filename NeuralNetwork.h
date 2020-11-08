#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <initializer_list>
#include <iostream>
#include <random>
#include <eigen3/Eigen/Core>
#include <fmt/core.h>
#include "random.h"

template<typename T>
inline auto sigmoid(const T& x)
{
	if constexpr(std::is_base_of_v<Eigen::DenseBase<T>, T>)
	{
		return 1 / (1 + Eigen::exp(-x.array()));
	}
	else
	{
		return 1 / (1 + std::exp(-x));
	}
}

template<typename T>
inline auto swish(const T& x)
{
	if constexpr(std::is_base_of_v<Eigen::DenseBase<T>, T>)
	{
		return x.array() / (1 + Eigen::exp(-x.array()));
	}
	else
	{
		return x / (1 + std::exp(-x));
	}
}

template<typename T>
inline auto relu(const T& x)
{
	if constexpr(std::is_base_of_v<Eigen::DenseBase<T>, T>)
	{
		return (x.array() < 0).select(0, x);
	}
	else
	{
		return std::max(x, 0);
	}
}

template<typename T>
inline auto softmax(const T& x)
{
	if constexpr(std::is_base_of_v<Eigen::DenseBase<T>, T>)
	{
		return Eigen::exp(x) / Eigen::exp(x).sum();
	}
	else
	{
		return 0;
	}
}

struct NeuralNetwork
{
	std::vector<Eigen::MatrixXf> weights;

	NeuralNetwork() = default;
	NeuralNetwork(std::initializer_list<const uint32_t> sizes, const bool randomize = true);
	
	template<typename Distribution = std::uniform_real_distribution<double>>
	NeuralNetwork(std::initializer_list<const uint32_t> sizes, Distribution& dis);
	uint32_t layersCount() const noexcept;
	Eigen::VectorXf feedForward(const Eigen::VectorXf& input) const;
};

template<typename Distribution>
inline NeuralNetwork::NeuralNetwork(std::initializer_list<const uint32_t> sizes, Distribution& dis)
{
	weights.reserve(sizes.size() - 1);
	for (uint32_t i = 0; i < sizes.size() - 1; ++i)
	{
		weights.emplace_back(Eigen::MatrixXf::NullaryExpr(*(sizes.begin() + i), *(sizes.begin() + i + 1), [&](){return dis(random::random_generator);} ));
	}
}

#endif // NEURAL_NETWORK_H
