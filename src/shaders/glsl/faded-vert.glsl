// "adi_gpu_vulkan" crate - Licensed under the MIT LICENSE
//  * Copyright (c) 2018  Jeron A. Lau <jeron.lau@plopgrizzly.com>

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (binding = 0) uniform UniformBuffer {
	mat4 models_tfm; // The Models' Transform Matrix
	float alpha;
	int has_camera;
} uniforms;
layout (binding = 1) uniform Camera {
	mat4 matrix; // The Camera's Transform & Projection Matrix
} camera;
layout (binding = 2) uniform Fog {
	vec4 fog; // The fog color.
	vec2 range; // The range of fog (fog to far clip)
} fog;
layout (binding = 3) uniform sampler2D tex;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 texpos;

layout (location = 0) out vec4 texcoord;
layout (location = 1) out float z;

void main() {
	texcoord = vec4(texpos.xyz, texpos.w * uniforms.alpha);

	vec4 place = uniforms.models_tfm * vec4(pos.xyz, 1.0);

	if(uniforms.has_camera >= 1) {
		gl_Position = camera.matrix * place;
	} else {
		gl_Position = place;
	}

	z = length(gl_Position.xyz);
}
