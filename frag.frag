#version 450

layout(binding=0) uniform usampler2D grid;
in vec2 texture_coord;
out vec4 out_color;

void main(void)
{
	vec3[] colors = {vec3(0.1, 0.1, 0.1), vec3(0.8, 0.8, 0.8), vec3(0.8, 0.4, 0.4), vec3(0.4, 0.4, 0.8), vec3(0.2, 0.2, 0.9)};
	out_color = vec4(colors[texture(grid, texture_coord).x], 1.0);
}
