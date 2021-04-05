#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(push_constant) uniform PushConstants {
    mat4 perspective_view;
    mat4 transform;
};

layout(location = 0) out vec2 out_uv;

void main() {
    out_uv = in_uv;

    gl_Position = perspective_view * transform * vec4(in_position, 1.0);
}
