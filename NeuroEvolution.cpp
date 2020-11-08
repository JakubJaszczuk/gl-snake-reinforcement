#include "NeuroEvolution.h"

uint32_t NeuroEvolution::tournament(const std::vector<double>& fitnesses, const uint32_t t_size) noexcept
{
	std::uniform_int_distribution<std::uint32_t> dis(0, fitnesses.size() - 1);
	auto best = dis(random::random_generator);
	double best_score = fitnesses[best];
	for(uint32_t i = 0; i < t_size - 1; ++i)
	{
		const auto selected = dis(random::random_generator);
		const double selected_score = fitnesses[selected];
		if(selected_score > best_score)
		{
			best = selected;
			best_score = selected_score;
		}
	}
	return best;
}

void NeuroEvolution::mutate(NeuralNetwork& nn)
{
	static std::uniform_real_distribution dis(-0.4f, 0.4f);
	for(auto& m : nn.weights)
	{
		for(uint32_t i = 0; i < m.size(); ++i)
		{
			m.data()[i] += dis(random::random_generator);
		}
	}
}

NeuralNetwork NeuroEvolution::cross(const NeuralNetwork& nn1, const NeuralNetwork& nn2)
{
	assert(nn1.weights.size() == nn2.weights.size());
	std::bernoulli_distribution dis;
	NeuralNetwork res(nn1);
	for(uint32_t i = 0; i < res.weights.size(); ++i)
	{
		const auto size = res.weights[i].size();
		for(uint32_t j = 0; j < size; ++j)
		{
			// Select weight from nn1 or nn2
			res.weights[i](j) = dis(random::random_generator) ? nn1.weights[i](j) : nn2.weights[i](j);
			//res.weights[i](j) = (nn1.weights[i](j) + nn2.weights[i](j)) / 2;
		}
	}
	return res;
}

NeuralNetwork NeuroEvolution::neuro_evolution(SnakeData& problem, const uint32_t iterations, const uint32_t pop_size, const float prob_mut, const float prob_cross, const uint32_t t_size, const uint32_t sim_time)
{
	std::uniform_real_distribution<double> prob(0.0, 1.0);
	std::uniform_int_distribution<uint32_t> pos(1, problem.data.rows()-2);
	// Vector fitnessów - im mniej tym lepiej
	std::vector<NeuralNetwork> population;
	population.reserve(pop_size);
	for(uint32_t i = 0; i < pop_size; ++i)
	{
		population.push_back({10, 3});
	}
	std::vector<NeuralNetwork> new_population(pop_size);
	std::vector<double> fitnesses(pop_size);
	for(uint64_t i = 0; i < iterations; ++i)
	{
		// Obliczenie fitnessów
		for(uint32_t iter = 0; iter < pop_size; ++iter)
		{
			SnakeNN snake(pos(random::random_generator), pos(random::random_generator), population[iter]);
			uint32_t step = 0;
			for(; step < sim_time && problem.step(snake); ++step);
			fitnesses[iter] = snake.score;
		}
		// Ewolucja właściwa
		for(uint32_t iter = 0; iter < pop_size; ++iter)
		{
			// Selekcja
			const auto selected_indx1 = NeuroEvolution::tournament(fitnesses, t_size);
			const auto selected_indx2 = NeuroEvolution::tournament(fitnesses, t_size);
			// Crossover
			NeuralNetwork new_nn = population[selected_indx1];
			if(prob(random::random_generator) < prob_cross)
			{
				new_nn = NeuroEvolution::cross(population[selected_indx1], population[selected_indx2]);
			}
			// Mutacja s1
			if(prob(random::random_generator) < prob_mut)
			{
				NeuroEvolution::mutate(new_nn);
			}
			// Dodaj do nowej populacji
			new_population[iter] = std::move(new_nn);
		}
		// Zamień populacje
		std::swap(population, new_population);
		if(!(i%100)){
			fmt::print("Best score: {}\n", *std::max_element(std::begin(fitnesses), std::end(fitnesses)));
		}
	}
	//fmt::print("\n{}\n", fitnesses);
	const auto index = std::max_element(std::begin(fitnesses), std::end(fitnesses));
	fmt::print("Selected Fitness: {}\n", *index);
	return population[index - std::begin(fitnesses)];
}

NeuralNetwork NeuroEvolution::neuro_evolution_steady(SnakeData& problem, const uint32_t iterations, const uint32_t pop_size, const float prob_mut, const float prob_cross, const uint32_t t_size, const uint32_t sim_time)
{
	std::uniform_real_distribution<double> prob(0.0, 1.0);
	std::uniform_int_distribution<uint32_t> pos(1, problem.data.rows()-2);
	// Vector fitnessów - im mniej tym lepiej
	std::vector<NeuralNetwork> population;
	population.reserve(pop_size);
	for(uint32_t i = 0; i < pop_size; ++i)
	{
		population.push_back({10, 3});
	}
	std::vector<double> fitnesses(pop_size);
	for(uint32_t iter = 0; iter < pop_size; ++iter)
	{
		SnakeNN snake(pos(random::random_generator), pos(random::random_generator), population[iter]);
		uint32_t step = 0;
		for(; step < sim_time && problem.step(snake); ++step);
		fitnesses[iter] = snake.score;
	}
	for(uint64_t i = 0; i < iterations; ++i)
	{
		// Selekcja
		const auto selected_indx1 = NeuroEvolution::tournament(fitnesses, t_size);
		const auto selected_indx2 = NeuroEvolution::tournament(fitnesses, t_size);
		// Crossover
		NeuralNetwork new_nn = NeuroEvolution::cross(population[selected_indx1], population[selected_indx2]);
		if(prob(random::random_generator) < prob_mut)
		{
			NeuroEvolution::mutate(new_nn);
		}

		const auto index = std::min_element(std::begin(fitnesses), std::end(fitnesses));

		SnakeNN snake(pos(random::random_generator), pos(random::random_generator), new_nn);
		uint32_t step = 0;
		for(; step < sim_time && problem.step(snake); ++step);
		*index = snake.score;

		population[index - std::begin(fitnesses)] = std::move(new_nn);

		if(!(i%4000)){
			fmt::print("Best score: {}\n", *std::max_element(std::begin(fitnesses), std::end(fitnesses)));
		}
	}
	//fmt::print("\n{}\n", fitnesses);
	const auto index = std::max_element(std::begin(fitnesses), std::end(fitnesses));
	fmt::print("Selected Fitness: {}\n", *index);
	return population[index - std::begin(fitnesses)];
}

NeuralNetwork NeuroEvolution::cross_entropy(SnakeData& problem, const uint32_t iterations, const uint32_t pop_size, const uint32_t elite_size, const double learn_rate, const uint32_t sim_time)
{
	std::uniform_int_distribution<uint32_t> pos(1, problem.data.rows()-2);
	NeuralNetwork global_nn({10, 3});
	std::vector<NeuralNetwork> population;
	std::vector<double> fitnesses;
	//std::vector<uint32_t> elite_indices;
	population.reserve(pop_size);
	fitnesses.reserve(pop_size);
	//elite_indices.reserve(elite_size);
	for(uint64_t i = 0; i < iterations; ++i)
	{
		// Generowanie sąsiadów
		for(uint32_t j = 0; j < pop_size; ++j)
		{
			auto& elem = population.emplace_back(global_nn);
			NeuroEvolution::mutate(elem);
			
			SnakeNN snake(pos(random::random_generator), pos(random::random_generator), elem);
			uint32_t step = 0;
			for(; step < sim_time && problem.step(snake); ++step);
			fitnesses.emplace_back(snake.score);
		}
		// Stworzenie elity
		for(uint32_t j = 0; j < elite_size; ++j)
		{
			const auto ptr = std::max_element(std::begin(fitnesses), std::end(fitnesses));
			const auto index = ptr - std::begin(fitnesses);

			for(uint32_t k = 0; k < global_nn.weights.size(); ++k)
			{
				//global_nn.weights[k] += learn_rate * (population[index].weights[k] / elite_size);
				global_nn.weights[k] = (population[index].weights[k] / elite_size);
			}

			//elite_indices.emplace_back(ptr - std::begin(fitnesses));
			*ptr = -1000;
		}

		//elite_indices.clear();
		if(!(i%100)){
			fmt::print("Best score: {}\n", *std::max_element(std::begin(fitnesses), std::end(fitnesses)));
		}
		population.clear();
		fitnesses.clear();
	}
	return global_nn;
}
