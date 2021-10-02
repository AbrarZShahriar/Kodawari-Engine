#pragma once

#include <glm/vec3.hpp>
#include <vector>
#include <vk_types.h>

struct Vertex {

  glm::vec3 position;
  glm::vec3 normal;
  glm::vec3 color;
};

struct Mesh {
  std::vector<Vertex> _vertices;

  AllocatedBuffer _vertexBuffer;
};
