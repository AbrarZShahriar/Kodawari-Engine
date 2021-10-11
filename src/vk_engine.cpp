﻿
#include "vk_engine.h"


#if defined(_WIN64)
// SDL
#include <SDL.h>
#include <SDL_vulkan.h>
#endif


#if defined(__linux__)
// SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#endif

#include <vk_initializers.h>

#include <vk_types.h>

// bootstrap library
#include "VkBootstrap.h"
#include <fstream>
#include <iostream>

// we want to immediately abort when there is an error. In normal engines this
// would give an error message to the user, or perform a dump of state.
using namespace std;
#define VK_CHECK(x) \
    do \
    { \
        VkResult err = x; \
        if(err) \
        { \
            std::cout << "Detected Vulkan error: " << err << std::endl; \
            abort(); \
        } \
    } while(0)
void
VulkanEngine::init()
{
    /*camXYZ[0] = 0.0;
    camXYZ[1] = -2;
    camXYZ[2] = 0.0;*/
    camFront = glm::vec3(0.0f, 0.0f, 0.0f);
    //camera pos
    camPos = {0.0f, 0.0f, -2.0f};


    //glm::vec3 camFront = glm::vec3(0.0f, 0.0f, 1.0f);
    camUp = glm::vec3(0.0f, 1.0f, 0.0f);
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow("Kodowari Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width, _windowExtent.height, window_flags);
    init_vulkan();

    // create the swapchain
    init_swapchain();
    init_commands();
    // everything went fine
    init_default_renderpass();

    init_framebuffers();

    //--- other code ---
    init_sync_structures();

    init_pipelines();
    load_meshes();
    init_scene();

    // everything went fine
    _isInitialized = true;
}
void
VulkanEngine::init_scene()
{
    RenderObject monkey;
    monkey.mesh = get_mesh("monkey");
    monkey.material = get_material("defaultmesh");
    monkey.transformMatrix = glm::mat4{1.0f};

    _renderables.push_back(monkey);


    _renderables.push_back(monkey);

    for(int x = -20; x <= 20; x++)
    {
        for(int y = -20; y <= 20; y++)
        {

            RenderObject tri;
            tri.mesh = get_mesh("monkey");
            tri.material = get_material("defaultmesh");
            glm::mat4 translation = glm::translate(glm::mat4{1.0}, glm::vec3(x, 0, y));
            glm::mat4 scale = glm::scale(glm::mat4{1.0}, glm::vec3(0.2, 0.2, 0.2));
            tri.transformMatrix = translation * scale;

            _renderables.push_back(tri);
        }
    }
}
void
VulkanEngine::load_meshes()
{
    // make the array 3 vertices long
    _triangleMesh._vertices.resize(3);

    // vertex positions
    _triangleMesh._vertices[0].position = {1.f, 1.f, 0.0f};
    _triangleMesh._vertices[1].position = {-1.f, 1.f, 0.0f};
    _triangleMesh._vertices[2].position = {0.f, -1.f, 0.0f};

    // vertex colors, all green
    _triangleMesh._vertices[0].color = {0.f, 1.f, 0.0f}; // pure green
    _triangleMesh._vertices[1].color = {0.f, 1.f, 0.0f}; // pure green
    _triangleMesh._vertices[2].color = {0.f, 1.f, 0.0f}; // pure green

    // we don't care about the vertex normals


    //load the monkey
    //_monkeyMesh.load_from_obj("../../assets/monkey_smooth.obj");
    _monkeyMesh.load_from_obj("../../assets/monkey_smooth.obj");

    upload_mesh(_triangleMesh);
    upload_mesh(_monkeyMesh);
    //note that we are copying them. Eventually we will delete the hardcoded _monkey and _triangle meshes, so it's no problem now.
    _meshes["monkey"] = _monkeyMesh;
    _meshes["triangle"] = _triangleMesh;
}

void
VulkanEngine::upload_mesh(Mesh &mesh)
{
    // allocate vertex buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    // this is the total size, in bytes, of the buffer we are allocating
    bufferInfo.size = mesh._vertices.size() * sizeof(Vertex);
    // this buffer is going to be used as a Vertex Buffer
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    // let the VMA library know that this data should be writeable by CPU, but
    // also readable by GPU
    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &mesh._vertexBuffer._buffer, &mesh._vertexBuffer._allocation, nullptr));

    // add the destruction of triangle mesh buffer to the deletion queue
    _mainDeletionQueue.push_function([=]() { vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation); });

    // copy vertex data
    void *data;
    vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);

    memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));

    vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);
}
void
VulkanEngine::init_vulkan()
{
    // nothing yet
    vkb::InstanceBuilder builder;

    // make the Vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Example Vulkan Application").request_validation_layers(true).require_api_version(1, 1, 0).use_default_debug_messenger().build();

    vkb::Instance vkb_inst = inst_ret.value();

    // store the instance
    _instance = vkb_inst.instance;
    // store the debug messenger
    _debug_messenger = vkb_inst.debug_messenger;

    // get the surface of the window we opened with SDL
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    // use vkbootstrap to select a GPU.
    // We want a GPU that can write to the SDL surface and supports Vulkan 1.1
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(1, 1).set_surface(_surface).select().value();

    // create the final Vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a Vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // use vkbootstrap to get a Graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    vmaCreateAllocator(&allocatorInfo, &_allocator);
}

void
VulkanEngine::init_swapchain()
{
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

    vkb::Swapchain vkbSwapchain = swapchainBuilder
                                        .use_default_format_selection()
                                        // use vsync present mode
                                        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                                        .set_desired_extent(_windowExtent.width, _windowExtent.height)
                                        .build()
                                        .value();

    // store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();

    _swapchainImageFormat = vkbSwapchain.image_format;

    //depth image size will match the window
    VkExtent3D depthImageExtent = {_windowExtent.width, _windowExtent.height, 1};
    //hardcoding the depth format to 32 bit float
    _depthFormat = VK_FORMAT_D32_SFLOAT;

    //the depth image will be an image with the format we selected and Depth Attachment usage flag
    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

    //for the depth image, we want to allocate it from GPU local memory
    VmaAllocationCreateInfo dimg_allocinfo = {};
    dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    //allocate and create the image
    vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr);

    //build an image-view for the depth image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

    //add to deletion queues


    _mainDeletionQueue.push_function([=]() { vkDestroySwapchainKHR(_device, _swapchain, nullptr); });
    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _depthImageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
    });
}
void
VulkanEngine::cleanup()
{
    if(_isInitialized)
    {

        // make sure the GPU has stopped doing its things
        vkWaitForFences(_device, 1, &_renderFence, true, 1000000000);

        _mainDeletionQueue.flush();

        vkDestroyCommandPool(_device, _commandPool, nullptr);
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);

        // destroy the main renderpass
        vkDestroyRenderPass(_device, _renderPass, nullptr);

        // destroy swapchain resources
        for(int i = 0; i < _framebuffers.size(); i++)
        {
            vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);

            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        }
        vmaDestroyAllocator(_allocator);

        vkDestroyDevice(_device, nullptr);
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);
    }
}
void
VulkanEngine::init_commands()
{
    // create a command pool for commands submitted to the graphics queue.
    // we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

    // allocate the default command buffer that we will use for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));

    _mainDeletionQueue.push_function([=]() { vkDestroyCommandPool(_device, _commandPool, nullptr); });
}

void
VulkanEngine::init_default_renderpass()
{
    // the renderpass will use this color attachment.
    VkAttachmentDescription color_attachment = {};
    // the attachment will have the format needed by the swapchain
    color_attachment.format = _swapchainImageFormat;
    // 1 sample, we won't be doing MSAA
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    // we Clear when this attachment is loaded
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // we keep the attachment stored when the renderpass ends
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // we don't care about stencil
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    // we don't know or care about the starting layout of the attachment
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // after the renderpass ends, the image has to be on a layout ready for
    // display
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference color_attachment_ref = {};
    // attachment number will index into the pAttachments array in the parent
    // renderpass itself
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depth_attachment = {};
    // Depth attachment
    depth_attachment.flags = 0;
    depth_attachment.format = _depthFormat;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attachment_ref = {};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // we are going to create 1 subpass, which is the minimum you can do
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    //hook the depth attachment into the subpass
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

    //array of 2 attachments, one for the color, and other for depth
    VkAttachmentDescription attachments[2] = {color_attachment, depth_attachment};

    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

    // connect the color attachment to the info
    render_pass_info.attachmentCount = 2;
    render_pass_info.pAttachments = &attachments[0];
    // connect the subpass to the info
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;

    VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));


    _mainDeletionQueue.push_function([=]() { vkDestroyRenderPass(_device, _renderPass, nullptr); });
}

void
VulkanEngine::init_framebuffers()
{
    // create the framebuffers for the swapchain images. This will connect the
    // render-pass to the images for rendering
    VkFramebufferCreateInfo fb_info = {};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;

    fb_info.renderPass = _renderPass;
    fb_info.attachmentCount = 1;
    fb_info.width = _windowExtent.width;
    fb_info.height = _windowExtent.height;
    fb_info.layers = 1;

    // grab how many images we have in the swapchain
    const uint32_t swapchain_imagecount = _swapchainImages.size();
    _framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

    // create framebuffers for each of the swapchain image views
    for(int i = 0; i < swapchain_imagecount; i++)
    {
        VkImageView attachments[2];
        attachments[0] = _swapchainImageViews[i];
        attachments[1] = _depthImageView;
        fb_info.pAttachments = attachments;
        fb_info.attachmentCount = 2;
        VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        });
    }
}

void
VulkanEngine::init_sync_structures()
{
    // create synchronization structures

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = nullptr;

    // we want to create the fence with the Create Signaled flag, so we can wait
    // on it before using it on a GPU command (for the first frame)
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

    // enqueue the destruction of the fence
    _mainDeletionQueue.push_function([=]() { vkDestroyFence(_device, _renderFence, nullptr); });

    // for the semaphores we don't need any flags
    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = nullptr;
    semaphoreCreateInfo.flags = 0;

    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));

    // enqueue the destruction of semaphores
    _mainDeletionQueue.push_function([=]() {
        vkDestroySemaphore(_device, _presentSemaphore, nullptr);
        vkDestroySemaphore(_device, _renderSemaphore, nullptr);
    });
}

void
VulkanEngine::draw()
{
    // wait until the GPU has finished rendering the last frame. Timeout of 1
    // second
    VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &_renderFence));

    // request image from the swapchain, one second timeout
    uint32_t swapchainImageIndex;
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, _presentSemaphore, nullptr, &swapchainImageIndex));

    // now that we are sure that the commands finished executing, we can safely
    // reset the command buffer to begin recording again.
    VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

    // naming it cmd for shorter writing
    VkCommandBuffer cmd = _mainCommandBuffer;

    // begin the command buffer recording. We will use this command buffer exactly
    // once, so we want to let Vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.pNext = nullptr;

    cmdBeginInfo.pInheritanceInfo = nullptr;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // make a clear-color from frame number. This will flash with a 120*pi frame
    // period.
    VkClearValue clearValue;
    float flash = abs(sin(_frameNumber / 120.f));
    clearValue.color = {{0.0f, 0.0f, flash, 1.0f}};
    //clear depth at 1
    VkClearValue depthClear;
    depthClear.depthStencil.depth = 1.f;

    // start the main renderpass.
    // We will use the clear color from above, and the framebuffer of the index
    // the swapchain gave us
    VkRenderPassBeginInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.pNext = nullptr;

    rpInfo.renderPass = _renderPass;
    rpInfo.renderArea.offset.x = 0;
    rpInfo.renderArea.offset.y = 0;
    rpInfo.renderArea.extent = _windowExtent;
    rpInfo.framebuffer = _framebuffers[swapchainImageIndex];

    // connect clear values
    rpInfo.clearValueCount = 2;
    VkClearValue clearValues[] = {clearValue, depthClear};
    rpInfo.pClearValues = &clearValues[0];

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);


    // once we start adding rendering commands, they will go here

    draw_objects(cmd, _renderables.data(), _renderables.size());


    // finalize the render pass
    vkCmdEndRenderPass(cmd);
    // finalize the command buffer (we can no longer add commands, but it can now
    // be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));

    // prepare the submission to the queue.
    // we want to wait on the _presentSemaphore, as that semaphore is signaled
    // when the swapchain is ready we will signal the _renderSemaphore, to signal
    // that rendering has finished

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = nullptr;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    submit.pWaitDstStageMask = &waitStage;

    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &_presentSemaphore;

    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &_renderSemaphore;

    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    // submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

    // this will put the image we just rendered into the visible window.
    // we want to wait on the _renderSemaphore for that,
    // as it's necessary that drawing commands have finished before the image is
    // displayed to the user
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;

    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &_renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

    // increase the number of frames drawn
    _frameNumber++;
}
bool
VulkanEngine::load_shader_module(const char *filePath, VkShaderModule *outShaderModule)
{
    // open the file. With cursor at the end
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if(!file.is_open())
    {
        return false;
    }

    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    size_t fileSize = (size_t)file.tellg();

    // spirv expects the buffer to be on uint32, so make sure to reserve an int
    // vector big enough for the entire file
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    // put file cursor at beginning
    file.seekg(0);

    // load the entire file into the buffer
    file.read((char *)buffer.data(), fileSize);

    // now that the file is loaded into the buffer, we can close it
    file.close();

    // create a new shader module, using the buffer we loaded
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;

    // codeSize has to be in bytes, so multiply the ints in the buffer by size of
    // int to know the real size of the buffer
    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    // check that the creation goes well.
    VkShaderModule shaderModule;
    if(vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        return false;
    }
    *outShaderModule = shaderModule;
    return true;
}

void
VulkanEngine::init_pipelines()
{
    //we start from just the default empty pipeline layout info
    VkPipelineLayoutCreateInfo mesh_pipeline_layout_info = vkinit::pipeline_layout_create_info();

    //setup push constants
    VkPushConstantRange push_constant;
    //this push constant range starts at the beginning
    push_constant.offset = 0;
    //this push constant range takes up the size of a MeshPushConstants struct
    push_constant.size = sizeof(MeshPushConstants);
    //this push constant range is accessible only in the vertex shader
    push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
    mesh_pipeline_layout_info.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info, nullptr, &_meshPipelineLayout));


    VkShaderModule triangleFragShader;
    if(!load_shader_module("../../shaders/colored_triangle.frag.spv", &triangleFragShader))
    {
        std::cout << "Error when building the triangle fragment shader module" << std::endl;
    }
    else
    {
        std::cout << "Triangle fragment shader successfully loaded" << std::endl;
    }

    VkShaderModule triangleVertexShader;
    if(!load_shader_module("../../shaders/colored_triangle.vert.spv", &triangleVertexShader))
    {
        std::cout << "Error when building the triangle vertex shader module" << std::endl;
    }
    else
    {
        std::cout << "Triangle vertex shader successfully loaded" << std::endl;
    }

    // compile red triangle modules
    VkShaderModule redTriangleFragShader;
    if(!load_shader_module("../../shaders/triangle.frag.spv", &redTriangleFragShader))
    {
        std::cout << "Error when building the triangle fragment shader module" << std::endl;
    }
    else
    {
        std::cout << "Red Triangle fragment shader successfully loaded" << std::endl;
    }

    VkShaderModule redTriangleVertShader;
    if(!load_shader_module("../../shaders/triangle.vert.spv", &redTriangleVertShader))
    {
        std::cout << "Error when building the triangle vertex shader module" << std::endl;
    }
    else
    {
        std::cout << "Red Triangle vertex shader successfully loaded" << std::endl;
    }

    // shader module loading

    // build the pipeline layout that controls the inputs/outputs of the shader
    // we are not using descriptor sets or other systems yet, so no need to use
    // anything other than empty default
    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));

    // layout and shader modules creation

    // build the stage-create-info for both vertex and fragment stages. This lets
    // the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;

    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, triangleVertexShader));

    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

    // vertex input controls how to read vertices from vertex buffers. We aren't
    // using it yet
    pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();

    // input assembly is the configuration for drawing triangle lists, strips, or
    // individual points. we are just going to draw triangle list
    pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // build viewport and scissor from the swapchain extents
    pipelineBuilder._viewport.x = 0.0f;
    pipelineBuilder._viewport.y = 0.0f;
    pipelineBuilder._viewport.width = (float)_windowExtent.width;
    pipelineBuilder._viewport.height = (float)_windowExtent.height;
    pipelineBuilder._viewport.minDepth = 0.0f;
    pipelineBuilder._viewport.maxDepth = 1.0f;

    pipelineBuilder._scissor.offset = {0, 0};
    pipelineBuilder._scissor.extent = _windowExtent;

    // configure the rasterizer to draw filled triangles
    pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

    // we don't use multisampling, so just run the default one
    pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

    // a single blend attachment with no blending and writing to RGBA
    pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();

    // use the triangle layout we created
    pipelineBuilder._pipelineLayout = _trianglePipelineLayout;
    //default depthtesting
    pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);
    // finally build the pipeline
    _trianglePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

    // clear the shader stages for the builder
    pipelineBuilder._shaderStages.clear();

    // add the other shaders
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, redTriangleVertShader));

    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, redTriangleFragShader));

    // build the red triangle pipeline
    _redTrianglePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);


    //build the mesh pipeline

    VertexInputDescription vertexDescription = Vertex::get_vertex_description();

    //connect the pipeline builder vertex input info to the one we get from Vertex
    pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
    pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

    pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
    pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

    //clear the shader stages for the builder
    pipelineBuilder._shaderStages.clear();

    //compile mesh vertex shader


    VkShaderModule meshVertShader;
    if(!load_shader_module("../../shaders/tri_mesh.vert.spv", &meshVertShader))
    {
        std::cout << "Error when building the triangle vertex shader module" << std::endl;
    }
    else
    {
        std::cout << "tri_mesh vertex shader successfully loaded" << std::endl;
    }

    //add the other shaders
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

    //make sure that triangleFragShader is holding the compiled colored_triangle.frag
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

    //build the mesh triangle pipeline
    _meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);


    //push consts
    pipelineBuilder._pipelineLayout = _meshPipelineLayout;

    _meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

    create_material(_meshPipeline, _meshPipelineLayout, "defaultmesh");


    //deleting all of the vulkan shaders
    vkDestroyShaderModule(_device, meshVertShader, nullptr);
    vkDestroyShaderModule(_device, redTriangleVertShader, nullptr);
    vkDestroyShaderModule(_device, redTriangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);


    _mainDeletionQueue.push_function([=]() {
        // destroy the 2 pipelines we have created
        vkDestroyPipeline(_device, _redTrianglePipeline, nullptr);
        vkDestroyPipeline(_device, _trianglePipeline, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);

        // destroy the pipeline layout that they use
        vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
    });
}
VkPipeline
PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass)
{
    // make viewport state from our stored viewport and scissor.
    // at the moment we won't support multiple viewports or scissors
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pNext = nullptr;

    viewportState.viewportCount = 1;
    viewportState.pViewports = &_viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &_scissor;

    // setup dummy color blending. We aren't using transparent objects yet
    // the blending is just "no blend", but we do write to the color attachment
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.pNext = nullptr;

    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &_colorBlendAttachment;

    // ... other code ...

    // build the actual pipeline
    // we now use all of the info structs we have been writing into into this one
    // to create the pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;

    pipelineInfo.stageCount = _shaderStages.size();
    pipelineInfo.pStages = _shaderStages.data();
    pipelineInfo.pVertexInputState = &_vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &_inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &_rasterizer;
    pipelineInfo.pMultisampleState = &_multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = _pipelineLayout;
    pipelineInfo.renderPass = pass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &_depthStencil;

    // it's easy to error out on create graphics pipeline, so we handle it a bit
    // better than the common VK_CHECK case
    VkPipeline newPipeline;
    if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS)
    {
        std::cout << "failed to create pipeline\n";
        return VK_NULL_HANDLE; // failed to create graphics pipeline
    }
    else
    {
        return newPipeline;
    }
}

glm::vec3
VulkanEngine::polarVector(float p, float y)
{
    // this form is already normalized
    return glm::vec3(std::cos(y) * std::cos(p), std::sin(p), std::sin(y) * std::cos(p));
}

// clamp pitch to [-89, 89]
float
VulkanEngine::clampPitch(float p)
{
    return p > 89.0f ? 89.0f : (p < -89.0f ? -89.0f : p);
}

// clamp yaw to [-180, 180] to reduce floating point inaccuracy
float
VulkanEngine::clampYaw(float y)
{
    float temp = (y + 180.0f) / 360.0f;
    return y - ((int)temp - (temp < 0.0f ? 1 : 0)) * 360.0f;
}

Material *
VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string &name)
{
    Material mat;
    mat.pipeline = pipeline;
    mat.pipelineLayout = layout;
    _materials[name] = mat;
    return &_materials[name];
}
Material *
VulkanEngine::get_material(const std::string &name)
{
    //search for the object, and return nullptr if not found
    auto it = _materials.find(name);
    if(it == _materials.end())
    {
        return nullptr;
    }
    else
    {
        return &(*it).second;
    }
}
Mesh *
VulkanEngine::get_mesh(const std::string &name)
{
    auto it = _meshes.find(name);
    if(it == _meshes.end())
    {
        return nullptr;
    }
    else
    {
        return &(*it).second;
    }
}
void
VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject *first, int count)
{
    //make a model view matrix for rendering the object
    //camera view
    glm::mat4 view = view = glm::lookAt(camPos, camPos + camFront, camUp); //glm::translate(glm::mat4(1.f), camPos);

    //cam proj
    glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.f);
    projection[1][1] *= -1;


    /////////


    Mesh *lastMesh = nullptr;
    Material *lastMaterial = nullptr;
    for(int i = 0; i < count; i++)
    {
        RenderObject &object = first[i];

        //only bind the pipeline if it doesn't match with the already bound one
        if(object.material != lastMaterial)
        {

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
            lastMaterial = object.material;
        }


        glm::mat4 model = object.transformMatrix;
        //final render matrix, that we are calculating on the cpu
        glm::mat4 mesh_matrix = projection * view * model;

        MeshPushConstants constants;
        constants.render_matrix = mesh_matrix;

        //upload the mesh to the GPU via push constants
        vkCmdPushConstants(cmd, object.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

        //only bind the mesh if it's a different one from last bind
        if(object.mesh != lastMesh)
        {
            //bind the mesh vertex buffer with offset 0
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);
            lastMesh = object.mesh;
        }
        //we can now draw
        vkCmdDraw(cmd, object.mesh->_vertices.size(), 1, 0, 0);
    }
}
void
VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;
    //mouse stuff

    const float sensitivity = 0.1f;
    const float cameraSpeed = 0.1f; // adjust accordingly

#define CTR_X (_windowExtent.width / 2)
#define CTR_Y (_windowExtent.height / 2)
#define RESET_MOUSE SDL_WarpMouseInWindow(_window, CTR_X, CTR_Y)

    // call once at the start
    RESET_MOUSE;

    // keep outside the loop
    float pitch = 0.0f, yaw = 0.0f;


    //continuous-response keys


    // main loop
    while(!bQuit)
    {
        // Handle events on queue
        while(SDL_PollEvent(&e) != 0)
        {


            // close the window when user alt-f4s or clicks the X button
            if(e.type == SDL_QUIT)
            {
                bQuit = true;
            }
            else if(e.type == SDL_KEYDOWN)
            {
                if(e.key.keysym.sym == SDLK_SPACE)
                {
                    _selectedShader += 1;
                    if(_selectedShader > 1)
                    {
                        _selectedShader = 0;
                    }
                }
            }

            if(e.type == SDL_MOUSEMOTION)
            {
                float deltaX = (float)e.motion.x - CTR_X;
                float deltaY = (float)e.motion.y - CTR_Y;

                yaw = clampYaw(yaw + sensitivity * deltaX);
                pitch = clampPitch(pitch - sensitivity * deltaY);

                // assumes radians input
                camFront = polarVector(glm::radians(pitch), glm::radians(yaw));

                // reset *every time*
                RESET_MOUSE;
            }

            //    // move cam
            //    if(e.type == SDL_KEYDOWN)
            //    {
            //        if(e.key.keysym.sym == SDLK_a)
            //        {
            //            camXYZ[0] += 0.1;
            //        }
            //        if(e.key.keysym.sym == SDLK_d)
            //        {
            //            camXYZ[0] -= 0.11;
            //        }

            //        if(e.key.keysym.sym == SDLK_w)
            //        {
            //            camXYZ[1] += 0.1;
            //        }
            //        if(e.key.keysym.sym == SDLK_s)
            //        {
            //            camXYZ[1] -= 0.11;
            //        }
            //    }
        }
        const Uint8 *keystate = SDL_GetKeyboardState(NULL);

        if(keystate[SDL_SCANCODE_A])
        {
            camPos -= glm::normalize(glm::cross(camFront, camUp)) * cameraSpeed;
        }
        if(keystate[SDL_SCANCODE_D])
        {
            camPos += glm::normalize(glm::cross(camFront, camUp)) * cameraSpeed;
        }
        if(keystate[SDL_SCANCODE_W])
        {
            camPos += cameraSpeed * camFront;
        }
        if(keystate[SDL_SCANCODE_S])
        {
            camPos -= cameraSpeed * camFront;
        }

        draw();
    }
}
