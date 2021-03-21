#version 450

layout(location = 0) out vec3 out_colour;

void main() {
    switch (gl_VertexIndex) {
        case 0:
            out_colour = vec3(1.0, 0.0, 0.0);
            gl_Position = vec4(-0.5, -0.5, 0.0, 1.0);
            break;
        case 1:
            out_colour = vec3(0.0, 1.0, 0.0);
            gl_Position = vec4(-0.75, 0.5, 0.0, 1.0);
            break;
        case 2:
            out_colour = vec3(0.0, 0.0, 1.0);
            gl_Position = vec4(-0.25, 0.5, 0.0, 1.0);
            break;

    }
}
