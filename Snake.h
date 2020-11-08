#ifndef SNAKE_H
#define SNAKE_H

#include <cstdint>
#include <list>
#include <vector>
#include <random>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <eigen3/Eigen/Core>
#include "NeuralNetwork.h"
#include "random.h"

struct SnakeData;

struct Snake
{
	enum class Actions : char
	{
		FORWARD, LEFT, RIGHT
	};

	enum class Directions : char
	{
		UP, DOWN, LEFT, RIGHT, NONE
	};

	std::list<std::pair<int32_t, int32_t>> body;
	int32_t score = 0;

	Snake();
	Snake(uint32_t x, uint32_t y);
	decltype (body)::const_reference head() const;
	void move(uint32_t x, uint32_t y);
	void removeTail();
	Directions direction() const noexcept;
	bool selfCollissin();
	virtual Actions doDecision();
	virtual void doAction(const SnakeData& state);
	virtual void useCurrentState(const SnakeData& state);
};


struct SnakeData
{
	static constexpr uint32_t EMPTY = 0;
	static constexpr uint32_t WALL = 1;
	static constexpr uint32_t REWARD = 2;
	static constexpr uint32_t SNAKE = 3;
	static constexpr uint32_t SNAKE_HEAD = 4;

	Eigen::ArrayXXi data = Eigen::ArrayXXi(10, 10);
	std::pair<int32_t, int32_t> reward_location;

	SnakeData();
	SnakeData(const uint32_t width, const uint32_t length);

	void defaultGrid();
	void placeReward();
	void placeReward(const Snake& snake);
	bool collission(const Snake& snake) const;
	bool step(Snake& snake);
	Eigen::ArrayXXi flatData(const Snake& snake) const;
	Eigen::ArrayXXi flatDataDisplay(const Snake& snake) const;
};


struct SnakeNN final : Snake
{
	NeuralNetwork nn;
	Eigen::VectorXf inputs;
	bool print = false;

	SnakeNN();
	SnakeNN(uint32_t x, uint32_t y);
	SnakeNN(std::initializer_list<const uint32_t> sizes);
	SnakeNN(const NeuralNetwork& nn);
	SnakeNN(uint32_t x, uint32_t y, std::initializer_list<const uint32_t> sizes);
	SnakeNN(uint32_t x, uint32_t y, const NeuralNetwork& nn);
	
	Actions doDecision() override;
	void useCurrentState(const SnakeData& state) override;
	void doAction(const SnakeData& state) override;
};

#endif // SNAKE_H
