#version 450

layout(location = 0) in vec2 in_uv;

layout(set = 0, binding = 0) uniform sampler2D sampler_2d;

layout(location = 0) out vec4 out_colour;

void main() {
    out_colour = texture(sampler_2d, in_uv);
}
