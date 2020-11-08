#ifndef NEUROEVOLUTION_H
#define NEUROEVOLUTION_H

#include "fmt/ranges.h"
#include "random.h"
#include "NeuralNetwork.h"
#include "Snake.h"

namespace NeuroEvolution
{
	uint32_t tournament(const std::vector<double>& fitnesses, const uint32_t t_size) noexcept;
	void mutate(NeuralNetwork& nn);
	template<typename Distribution = std::uniform_real_distribution<double>>
	void mutate(NeuralNetwork& nn, Distribution& dis);
	NeuralNetwork cross(const NeuralNetwork& nn1, const NeuralNetwork& nn2);
	NeuralNetwork neuro_evolution(SnakeData& problem, const uint32_t iterations, const uint32_t pop_size, const float prob_mut, const float prob_cross, const uint32_t t_size, const uint32_t sim_time);
	NeuralNetwork neuro_evolution_steady(SnakeData& problem, const uint32_t iterations, const uint32_t pop_size, const float prob_mut, const float prob_cross, const uint32_t t_size, const uint32_t sim_time);
	NeuralNetwork cross_entropy(SnakeData& problem, const uint32_t iterations, const uint32_t pop_size, const uint32_t elite_size, const double learn_rate, const uint32_t sim_time);
};

template<typename Distribution>
inline void NeuroEvolution::mutate(NeuralNetwork& nn, Distribution& dis)
{
	for(auto& m : nn.weights)
	{
		for(uint32_t i = 0; i < m.size(); ++i)
		{
			m.data()[i] += dis(random::random_generator);
		}
	}
}

#endif // NEUROEVOLUTION_H
