#include <chrono>
#include <eigen3/Eigen/Core>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include "NeuralNetwork.h"
#include "Snake.h"
#include "NeuroEvolution.h"
#include "utils.h"

int main()
{
	SnakeData sd;
	std::uniform_int_distribution<uint32_t> pos(1, sd.data.rows()-2);
	//const auto nn = NeuroEvolution::neuro_evolution(sd, 1000, 300, 0.5, 0.5, 10, 1000);
	const auto nn = NeuroEvolution::neuro_evolution_steady(sd, 100000, 400, 0.5, 0.8, 10, 1000);
	SnakeNN snake(pos(random::random_generator), pos(random::random_generator), nn);
	snake.print = true;
	fmt::print("Final NN layers {}, weights:\n", nn.layersCount());
	for(const auto& layer : nn.weights) {
		fmt::print("{}\n\n", layer);
	}

	if(!glfwInit())
	{
		return 1;
	}

	int32_t res_width = 512;
	int32_t res_height = 512;

	glfwSetErrorCallback(utils::error_callback);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
	GLFWwindow* window = glfwCreateWindow(res_width, res_height, "Snake", NULL, NULL);

	if(!window)
	{
		glfwTerminate();
	}

	glfwSetFramebufferSizeCallback(window, utils::framebuffer_size_callback);
	glfwMakeContextCurrent(window);
	glDebugMessageCallback(utils::gl_message_callback, 0);  // Super fajna rzecz
	glfwSwapInterval(1);

	// Shader
	const auto vs = utils::loadShader("vert.vert", GL_VERTEX_SHADER);
	const auto fs = utils::loadShader("frag.frag", GL_FRAGMENT_SHADER);
	const auto program = utils::createProgram({vs, fs});
	glUseProgram(program);

	GLuint texture;
	glCreateTextures(GL_TEXTURE_2D, 1, &texture);
	glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTextureStorage2D(texture, 1, GL_R8UI, sd.data.rows(), sd.data.cols());
	glBindTextureUnit(0, texture);
	glTextureSubImage2D(texture, 0, 0, 0, sd.data.rows(), sd.data.cols(), GL_RED_INTEGER, GL_INT, sd.data.data());

	// VAO
	GLuint vao;
	glCreateVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// Images
	glBindImageTexture(0, texture, 0, false, 0, GL_READ_ONLY, GL_R8UI);

	auto last_simulation_time = std::chrono::steady_clock::now();
	bool exit = false;
	while(!glfwWindowShouldClose(window))
	{
		// Simulation
		auto time_now = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::milli> time_diff = time_now - last_simulation_time;
		if(time_diff > std::chrono::milliseconds(300))
		{
			last_simulation_time = time_now;
			if(!exit)
			{
				exit = !sd.step(snake);
				fmt::print("Score: {}\n", snake.score);
				if(exit) fmt::print("\n## EXIT ##\n{}\n", sd.flatData(snake));
			}
			Eigen::ArrayXXi map = sd.flatDataDisplay(snake);
			glTextureSubImage2D(texture, 0, 0, 0, map.rows(), map.cols(), GL_RED_INTEGER, GL_INT, map.data());
		}

		// Draw
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		// Events
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteProgram(program);
	glDeleteVertexArrays(1, &vao);

	glfwDestroyWindow(window);
	glfwTerminate();
	
	return 0;
}
