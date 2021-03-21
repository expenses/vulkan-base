#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_colour;

layout(push_constant) uniform PushConstants {
    mat4 perspective_view;
    mat4 cube_transform;
};

layout(location = 0) out vec3 out_colour;

void main() {
    out_colour = in_colour;

    gl_Position = perspective_view * cube_transform * vec4(in_position, 1.0);
}
