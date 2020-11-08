#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::initializer_list<const uint32_t> sizes, const bool randomize)
{
	weights.reserve(sizes.size() - 1);
	std::uniform_real_distribution dis(-1.0, 1.0);
	for (uint32_t i = 0; i < sizes.size() - 1; ++i)
	{
		if (!randomize)
		{
			weights.emplace_back(Eigen::MatrixXf::Zero(*(sizes.begin() + i), *(sizes.begin() + i + 1)));
		}
		else
		{
			weights.emplace_back(Eigen::MatrixXf::NullaryExpr(*(sizes.begin() + i), *(sizes.begin() + i + 1), [&](){return dis(random::random_generator);} ));
			//weights.emplace_back(Eigen::MatrixXf::Random(*(sizes.begin() + i), *(sizes.begin() + i + 1)));
		}
	}
}

uint32_t NeuralNetwork::layersCount() const noexcept
{
	return weights.size() + 1;
}

Eigen::VectorXf NeuralNetwork::feedForward(const Eigen::VectorXf &input) const
{
	assert(input.rows() == weights.front().rows());
	Eigen::RowVectorXf res = input;
	for (const auto &mat : weights)
	{
		res = sigmoid(res * mat);
	}
	return res;
}
