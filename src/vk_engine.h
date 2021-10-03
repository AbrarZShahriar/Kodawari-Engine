// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <vk_types.h>

#include <functional>

#include <deque>
#include <vk_mesh.h>

//add the include for glm to get matrices
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>


struct DeletionQueue
{
    std::deque<std::function<void()>> deletors;

    void
    push_function(std::function<void()> &&function)
    {
        deletors.push_back(function);
    }

    void
    flush()
    {
        // reverse iterate the deletion queue to execute all the functions
        for(auto it = deletors.rbegin(); it != deletors.rend(); it++)
        {
            (*it)(); // call the function
        }

        deletors.clear();
    }
};


struct MeshPushConstants
{
    glm::vec4 data;
    glm::mat4 render_matrix;
};

class VulkanEngine
{
  public:
    float camXYZ[3];
    glm::vec3 camFront;
    //camera pos
    glm::vec3 camPos;


    //glm::vec3 camFront = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 camUp;
    // --- omitted ---
    VkInstance _instance;                      // Vulkan library handle
    VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
    VkPhysicalDevice _chosenGPU;               // GPU chosen as the default device
    VkDevice _device;                          // Vulkan device for commands
    VkSurfaceKHR _surface;                     // Vulkan window surface
                                               // --- other code ---
    VkSwapchainKHR _swapchain;                 // from other articles

    // image format expected by the windowing system
    VkFormat _swapchainImageFormat;

    // array of images from the swapchain
    std::vector<VkImage> _swapchainImages;

    // array of image-views from the swapchain
    std::vector<VkImageView> _swapchainImageViews;

    // --- other code ---
    VkQueue _graphicsQueue;        // queue we will submit to
    uint32_t _graphicsQueueFamily; // family of that queue

    VkCommandPool _commandPool;         // the command pool for our commands
    VkCommandBuffer _mainCommandBuffer; // the buffer we will record into

    //--- other code ---
    VkRenderPass _renderPass;

    std::vector<VkFramebuffer> _framebuffers;

    //--- other code ---
    VkSemaphore _presentSemaphore, _renderSemaphore;
    VkFence _renderFence;

    VkPipeline _meshPipeline;
    Mesh _triangleMesh;

    // loads a shader module from a spir-v file. Returns false if it errors
    bool
    load_shader_module(const char *filePath, VkShaderModule *outShaderModule);

    bool _isInitialized{false};
    int _frameNumber{0};
    int _selectedShader{0};
    VmaAllocator _allocator; // vma lib allocator

    VkExtent2D _windowExtent{1700, 900};

    struct SDL_Window *_window{nullptr};


    //cam struff

    glm::vec3
    polarVector(float p, float y);

    // clamp pitch to [-89, 89]
    float
    clampPitch(float p);
    // clamp yaw to [-180, 180] to reduce floating point inaccuracy
    float
    clampYaw(float y);


    // initializes everything in the engine
    void
    init();

    // shuts down the engine
    void
    cleanup();

    // draw loop
    void
    draw();

    // run main loop
    void
    run();

    // ... other stuff ....

    VkPipelineLayout _trianglePipelineLayout;
    VkPipelineLayout _meshPipelineLayout;

    // ... other objects
    VkPipeline _trianglePipeline;
    VkPipeline _redTrianglePipeline;
    DeletionQueue _mainDeletionQueue;

  private:
    void
    init_vulkan();
    // --- other code ---
    void
    init_swapchain();
    //----- other code----
    void
    init_commands();
    //--- other code ---
    void
    init_default_renderpass();

    void
    init_framebuffers();
    //--- other code ---
    void
    init_sync_structures();
    void
    init_pipelines();

    void
    load_meshes();

    void
    upload_mesh(Mesh &mesh);
};

class PipelineBuilder
{
  public:
    std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
    VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
    VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
    VkViewport _viewport;
    VkRect2D _scissor;
    VkPipelineRasterizationStateCreateInfo _rasterizer;
    VkPipelineColorBlendAttachmentState _colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo _multisampling;
    VkPipelineLayout _pipelineLayout;

    VkPipeline
    build_pipeline(VkDevice device, VkRenderPass pass);
};
