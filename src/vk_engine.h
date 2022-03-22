// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <unordered_map>
#include <vk_types.h>

#include <functional>

#include <deque>
#include <vk_mesh.h>

#include <tower_defense.h>

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
struct UploadContext
{
    VkFence _uploadFence;
    VkCommandPool _commandPool;
    VkCommandBuffer _commandBuffer;
};

struct MeshPushConstants
{
    glm::vec4 data;
    glm::mat4 render_matrix;
};
struct Material
{
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct RenderObject
{
    Mesh *mesh;

    Material *material;

    glm::mat4 transformMatrix;

    int id=0;
    


};
struct GPUCameraData
{
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 viewproj;
};
struct FrameData
{
    VkSemaphore _presentSemaphore, _renderSemaphore;
    VkFence _renderFence;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    //buffer that holds a single GPUCameraData to use when rendering
    AllocatedBuffer cameraBuffer;

    VkDescriptorSet globalDescriptor;
    AllocatedBuffer objectBuffer;
    VkDescriptorSet objectDescriptor;
};

struct GPUObjectData
{
    glm::mat4 modelMatrix;
};
//number of frames to overlap when rendering
constexpr unsigned int FRAME_OVERLAP = 2;

struct GPUSceneData
{
    glm::vec4 fogColor;     // w is for exponent
    glm::vec4 fogDistances; //x for min, y for max, zw unused.
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; //w for sun power
    glm::vec4 sunlightColor;
};
class VulkanEngine
{
  public:
    entity_megastruct EntityMegastruct[1000];
    game_controller GameControllerPlayer;
    game_controller GameControllerEnemy;
    game_state GameState;
   

    UploadContext _uploadContext;

    void
    immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);

    float camXYZ[3];
    //glm::vec3 camFront;
    //camera pos
    glm::vec3 camPos = glm::vec3(-10.0f, 9.0f, 3.0f);
    


    glm::vec3 camFront = glm::vec3(64.0f, -87.0f, 0.0f);
    glm::vec3 camUp;


    // ---  ---
    VkPhysicalDeviceProperties _gpuProperties;
    VkDescriptorSetLayout _globalSetLayout;
    VkDescriptorSetLayout _objectSetLayout;
    VkDescriptorPool _descriptorPool;
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
    VkImageView _depthImageView;
    AllocatedImage _depthImage;
    //the format for the depth image
    VkFormat _depthFormat;
    // --- other code ---
    VkQueue _graphicsQueue;        // queue we will submit to
    uint32_t _graphicsQueueFamily; // family of that queue

    //VkCommandPool _commandPool;         // the command pool for our commands
    //VkCommandBuffer _mainCommandBuffer; // the buffer we will record into

    //--- other code ---
    VkRenderPass _renderPass;

    std::vector<VkFramebuffer> _framebuffers;

    //--- other code ---
    //
    //
    GPUSceneData _sceneParameters;
    AllocatedBuffer _sceneParameterBuffer;
    //VkSemaphore _presentSemaphore, _renderSemaphore;
    //VkFence _renderFence;
    //frame storage
    FrameData _frames[FRAME_OVERLAP];
    VkPipeline _meshPipeline;
    Mesh _triangleMesh;
    Mesh _monkeyMesh;
    //default array of renderable objects
    std::vector<RenderObject> _renderables;
    std::unordered_map<std::string, Material> _materials;
    std::unordered_map<std::string, Mesh> _meshes;
    // loads a shader module from a spir-v file. Returns false if it errors
    bool
    load_shader_module(const char *filePath, VkShaderModule *outShaderModule);

    bool _isInitialized{false};
    int _frameNumber{0};
    int _selectedShader{0};
    VmaAllocator _allocator; // vma lib allocator

    VkExtent2D _windowExtent{1700, 900};

    struct SDL_Window *_window{nullptr};

    //create material and add it to the map
    Material *
    create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string &name);

    //returns nullptr if it can't be found
    Material *
    get_material(const std::string &name);

    //returns nullptr if it can't be found
    Mesh *
    get_mesh(const std::string &name);

    //our draw function
    void
    draw_objects(VkCommandBuffer cmd, RenderObject *first, int count);

    //cam struff

    size_t
    pad_uniform_buffer_size(size_t originalSize);

    glm::vec3
    polarVector(float p, float y);

    // clamp pitch to [-89, 89]
    float
    clampPitch(float p);
    // clamp yaw to [-180, 180] to reduce floating point inaccuracy
    float
    clampYaw(float y);

    AllocatedBuffer
    create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);


    // initializes everything in the engine
    void
    init();
    void
    init_imgui();
    void
    init_scene();
    void
    init_descriptors();

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
    //getter for the frame we are rendering to right now.
    FrameData &
    get_current_frame();
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
    init_pipelinesOLD();
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
    VkPipelineDepthStencilStateCreateInfo _depthStencil;
    VkPipeline
    build_pipeline(VkDevice device, VkRenderPass pass);
};
