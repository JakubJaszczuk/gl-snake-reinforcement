#include "Snake.h"

Snake::Snake() : body{{1, 0}, {0, 0}} {}

Snake::Snake(uint32_t x, uint32_t y) : body{{x, y}, {x-1, y}} {}

decltype (Snake::body)::const_reference Snake::head() const
{
	return body.front();
}

void Snake::move(uint32_t x, uint32_t y)
{
	body.emplace_front(x, y);
}

void Snake::removeTail()
{
	body.pop_back();
}

Snake::Directions Snake::direction() const noexcept
{
	auto x = body.front().first - (*std::next(body.begin())).first;
	auto y = body.front().second - (*std::next(body.begin())).second;
	if(y == 0 && x > 0) return Directions::RIGHT;
	if(y == 0 && x < 0) return Directions::LEFT;
	if(x == 0 && y > 0) return Directions::UP;
	if(x == 0 && y < 0) return Directions::DOWN;
	else return Directions::NONE;
}

bool Snake::selfCollissin()
{
	return std::find(std::next(std::begin(body)), std::end(body), body.front()) != std::end(body);
}

Snake::Actions Snake::doDecision()
{
	std::uniform_int_distribution<uint8_t> dis(0, 2);
	return static_cast<Snake::Actions>(dis(random::random_generator));
}

void Snake::doAction(const SnakeData& state)
{
	const auto action = doDecision();
	const auto dir = direction();
	int8_t dis_x = 0;
	int8_t dis_y = 0;
	if( (dir == Directions::UP && action == Actions::FORWARD) || 
		(dir == Directions::RIGHT && action == Actions::LEFT) || 
		(dir == Directions::LEFT && action == Actions::RIGHT)) dis_y = 1;
	else 
	if( (dir == Directions::DOWN && action == Actions::FORWARD) || 
		(dir == Directions::RIGHT && action == Actions::RIGHT) || 
		(dir == Directions::LEFT && action == Actions::LEFT)) dis_y = -1;
	else 
	if( (dir == Directions::UP && action == Actions::RIGHT) || 
		(dir == Directions::RIGHT && action == Actions::FORWARD) || 
		(dir == Directions::DOWN && action == Actions::LEFT)) dis_x = 1;
	else
	if( (dir == Directions::UP && action == Actions::LEFT) || 
		(dir == Directions::LEFT && action == Actions::FORWARD) || 
		(dir == Directions::DOWN && action == Actions::RIGHT)) dis_x = -1;

	const auto new_x = body.front().first + dis_x;
	const auto new_y = body.front().second + dis_y;
	move(new_x, new_y);
}

void Snake::useCurrentState(const SnakeData& state) {}

SnakeData::SnakeData() : data(10, 10)
{
	defaultGrid();
	placeReward();
}

SnakeData::SnakeData(const uint32_t width, const uint32_t length) : data(width, length)
{
	defaultGrid();
	placeReward();
}

void SnakeData::defaultGrid()
{
	data = 0;
	data.row(0) = 1;
	data.row(data.rows()-1) = 1;
	data.col(0) = 1;
	data.col(data.cols()-1) = 1;
}

void SnakeData::placeReward()
{
	std::uniform_int_distribution<Eigen::Index> dis_x(1, data.cols()-2);
	std::uniform_int_distribution<Eigen::Index> dis_y(1, data.rows()-2);
	reward_location = {dis_x(random::random_generator), dis_y(random::random_generator)};
	//data(dis_x(random::random_generator), dis_y(random::random_generator)) = REWARD;
}

void SnakeData::placeReward(const Snake& snake)
{
	std::uniform_int_distribution<Eigen::Index> dis_x(1, data.cols()-2);
	std::uniform_int_distribution<Eigen::Index> dis_y(1, data.rows()-2);
	do
	{
		reward_location = {dis_x(random::random_generator), dis_y(random::random_generator)};
	}
	while(std::find(std::begin(snake.body), std::end(snake.body), reward_location) != std::end(snake.body));
}

bool SnakeData::collission(const Snake& snake) const
{
	const auto& [x, y] = snake.body.front();
	return data(x, y) == WALL;
}

bool SnakeData::step(Snake& snake)
{
	snake.useCurrentState(*this);
	snake.doAction(*this);
	//const auto& [x, y] = snake->body.front();
	if(snake.body.front() == reward_location)
	{
		placeReward(snake);
		++(snake.score);
	}
	else
		snake.removeTail();
	const bool next = !(collission(snake) || snake.selfCollissin());
	//if(next == false) snake.score = 0;
	return next;
}

Eigen::ArrayXXi SnakeData::flatData(const Snake& snake) const
{
	Eigen::ArrayXXi res = data;
	for(const auto& e : snake.body)
	{
		res(e.first, e.second) = WALL;
	}
	return res;
}

Eigen::ArrayXXi SnakeData::flatDataDisplay(const Snake& snake) const
{
	Eigen::ArrayXXi res = data;
	res(reward_location.first, reward_location.second) = REWARD;
	for(const auto& e : snake.body)
	{
		res(e.first, e.second) = SNAKE;
	}
	res(snake.head().first, snake.head().second) = SNAKE_HEAD;
	return res;
}


SnakeNN::SnakeNN() : Snake() {}

SnakeNN::SnakeNN(uint32_t x, uint32_t y) : Snake(x, y) {}

SnakeNN::SnakeNN(std::initializer_list<const uint32_t> sizes) : 
	nn(sizes), 
	inputs(*std::begin(sizes)) {}

SnakeNN::SnakeNN(const NeuralNetwork& nn) : nn(nn), inputs(nn.weights[0].rows()) {}

SnakeNN::SnakeNN(uint32_t x, uint32_t y, std::initializer_list<const uint32_t> sizes) : 
	Snake(x, y), 
	nn(sizes), 
	inputs(*std::begin(sizes)) {}

SnakeNN::SnakeNN(uint32_t x, uint32_t y, const NeuralNetwork& nn) : 
	Snake(x, y), 
	nn(nn), 
	inputs(nn.weights[0].rows()) {}

Snake::Actions SnakeNN::doDecision()
{
	auto output = nn.feedForward(inputs);
	decltype(output)::Index ind;
	output.maxCoeff(&ind);
	if(print) fmt::print("Probabilities: {}\n", output.transpose());
	return static_cast<Snake::Actions>(ind);
}

void SnakeNN::useCurrentState(const SnakeData& state)
{
	const auto dir = direction();
	const auto [sx, sy] = head();
	auto x = sx - state.reward_location.first;
	auto y = sy - state.reward_location.second;
	if (dir == Snake::Directions::DOWN)
	{
		x = -x;
		y = -y;
	}
	else if (dir == Snake::Directions::RIGHT)
	{
		y = -y;
		std::swap(x, y);
	}
	else if (dir == Snake::Directions::LEFT)
	{
		x = -x;
		std::swap(x, y);
	}
	Eigen::ArrayXXi data = state.flatDataDisplay(*this);
	inputs << 
		//std::clamp(static_cast<double>(x), -0.9, 0.9), 
		//std::clamp(static_cast<double>(y), -0.9, 0.9), 
		x, 
		y, 
		data(sx+1, sy), 
		data(sx, sy+1), 
		data(sx-1, sy), 
		data(sx, sy-1),
		data(sx+1, sy+1), 
		data(sx-1, sy+1), 
		data(sx-1, sy-1), 
		data(sx+1, sy-1);
}

void SnakeNN::doAction(const SnakeData& state)
{
	const auto action = doDecision();
	const auto dir = direction();
	int8_t dis_x = 0;
	int8_t dis_y = 0;
	if( (dir == Directions::UP && action == Actions::FORWARD) || 
		(dir == Directions::RIGHT && action == Actions::LEFT) || 
		(dir == Directions::LEFT && action == Actions::RIGHT)) dis_y = 1;
	else 
	if( (dir == Directions::DOWN && action == Actions::FORWARD) || 
		(dir == Directions::RIGHT && action == Actions::RIGHT) || 
		(dir == Directions::LEFT && action == Actions::LEFT)) dis_y = -1;
	else 
	if( (dir == Directions::UP && action == Actions::RIGHT) || 
		(dir == Directions::RIGHT && action == Actions::FORWARD) || 
		(dir == Directions::DOWN && action == Actions::LEFT)) dis_x = 1;
	else
	if( (dir == Directions::UP && action == Actions::LEFT) || 
		(dir == Directions::LEFT && action == Actions::FORWARD) || 
		(dir == Directions::DOWN && action == Actions::RIGHT)) dis_x = -1;

	const auto x = body.front().first;
	const auto y = body.front().second;
	auto new_x = x + dis_x;
	auto new_y = y + dis_y;
	/*
	if(const auto data(state.flatData(*this)); data(new_x, new_y) == SnakeData::WALL)
	{
		if(print) fmt::print("CHEAT\n");
		if(data(x+1, y) == SnakeData::EMPTY) {new_x = x + 1; new_y = y;}
		else if(data(x-1, y) == SnakeData::EMPTY) {new_x = x - 1; new_y = y;}
		else if(data(x, y+1) == SnakeData::EMPTY) {new_x = x; new_y = y + 1;}
		else if(data(x, y-1) == SnakeData::EMPTY) {new_x = x; new_y = y - 1;}
	}
	*/
	move(new_x, new_y);
}
