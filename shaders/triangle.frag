//glsl version 4.5
#version 450

//in
layout (location = 0) in vec3 inColor;


//output write
layout (location = 0) out vec4 outFragColor;



void main()
{
	//return red
	outFragColor = vec4(1.f,0.f,0.f,1.0f);
}