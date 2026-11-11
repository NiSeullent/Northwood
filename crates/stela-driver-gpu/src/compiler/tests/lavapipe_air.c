/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Run the compiler fixture through Mesa's internal NIR stage path with
 * native Float16 disabled. The GPU writes the original host allocation. */
#include "../metal_abi.h"
#include "vulkan/util/vk_internal_exts.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
   if (!(condition)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      exit(EXIT_FAILURE); \
   } \
} while (0)

#define VK(call) do { \
   VkResult result = (call); \
   if (result != VK_SUCCESS) { \
      fprintf(stderr, "%s: %d\n", #call, result); \
      exit(EXIT_FAILURE); \
   } \
} while (0)

enum { table_stride = 2048, bytes = 16384 };

static const uint32_t half_cases[][4] = {
   {0x42ffe000, 0x3f802000, 0x38802000, 0x43002000}, /* compensated rounding */
   {0x3fc00000, 0x40000000, 0xbf800000, 0x40000000},
   {0x33800000, 0x40000000, 0x00000000, 0x34000000}, /* half denormal */
   {0x477fe000, 0x40000000, 0x00000000, 0x7f800000}, /* half overflow */
   {0x80000000, 0x3f800000, 0x80000000, 0x80000000}, /* signed zero */
};
static_assert(table_stride >= STELA_METAL_TABLE_BYTES, "argument table fits");

void stela_test_lavapipe_software_half(nir_shader *nir);
void stela_test_lavapipe_fp32_preserve(nir_shader *preserve, nir_shader *flush);
void stela_test_lavapipe_software_render(nir_shader *vertex, nir_shader *fragment);

struct lavapipe {
   VkInstance instance;
   VkPhysicalDevice physical;
   VkDevice device;
   VkQueue queue;
   uint32_t family;
   uint64_t host_alignment;
};

struct imported_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   unsigned char *mapping;
};

static uint32_t
memory_type(VkPhysicalDevice physical, uint32_t types, VkMemoryPropertyFlags required)
{
   VkPhysicalDeviceMemoryProperties memory;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory);
   for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
      if ((types & (UINT32_C(1) << i)) &&
          (memory.memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   CHECK(0 && "No compatible memory type");
   return 0;
}

static struct lavapipe
open_lavapipe(void)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance instance;
   VK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice physical;
   VK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   CHECK(count == 1);
   VkPhysicalDeviceDriverProperties driver = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
   };
   VkPhysicalDeviceExternalMemoryHostPropertiesEXT host = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
      .pNext = &driver,
   };
   VkPhysicalDeviceProperties2 properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &host,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties);
   CHECK(driver.driverID == VK_DRIVER_ID_MESA_LLVMPIPE);
   CHECK(properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU);
   CHECK(properties.properties.apiVersion >= VK_API_VERSION_1_3);
   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   CHECK(features.features.shaderInt16 && features.features.shaderInt64 && v12.bufferDeviceAddress);
   printf("CPU %s shaderFloat16=%u shaderInt16=%u host_alignment=%" PRIu64 "\n",
          driver.driverInfo, v12.shaderFloat16, features.features.shaderInt16,
          host.minImportedHostPointerAlignment);
   CHECK(!v12.shaderFloat16);
   uint32_t queues = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queues, NULL);
   CHECK(queues);
   VkQueueFamilyProperties *families = calloc(queues, sizeof(*families));
   CHECK(families);
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queues, families);
   uint32_t family = 0;
   const VkQueueFlags universal = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
   while (family < queues && (families[family].queueFlags & universal) != universal)
      family++;
   free(families);
   CHECK(family < queues);
   const float priority = 1.0f;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
      .queueCount = 1, .pQueuePriorities = &priority,
   };
   VkPhysicalDeviceVulkan13Features enabled13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .synchronization2 = VK_TRUE,
      .dynamicRendering = VK_TRUE,
   };
   VkPhysicalDeviceVulkan12Features enabled12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &enabled13,
      .bufferDeviceAddress = VK_TRUE,
   };
   const char *extensions[] = { VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME };
   const VkPhysicalDeviceFeatures enabled = { .shaderInt16 = VK_TRUE, .shaderInt64 = VK_TRUE };
   VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &enabled12, .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info, .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions, .pEnabledFeatures = &enabled,
   };
   VkDevice device;
   VK(vkCreateDevice(physical, &device_info, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   return (struct lavapipe){instance, physical, device, queue, family, host.minImportedHostPointerAlignment};
}

static struct imported_buffer
import_buffer(struct lavapipe lvp, VkBufferUsageFlags usage)
{
   VkDevice device = lvp.device;
   const VkExternalMemoryHandleTypeFlagBits handle =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
   const VkExternalMemoryBufferCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = handle,
   };
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &external, .size = bytes,
      .usage = usage,
   };
   VkBuffer buffer;
   VK(vkCreateBuffer(device, &buffer_info, NULL, &buffer));
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, buffer, &requirements);
   const uint64_t alignment = lvp.host_alignment;
   CHECK(alignment >= sizeof(void *) && !(alignment & (alignment - 1)) && alignment <= bytes);
   CHECK(requirements.size >= bytes && requirements.size <= SIZE_MAX - (alignment - 1));
   const size_t size = (requirements.size + alignment - 1) & ~(alignment - 1);
   unsigned char *mapping = aligned_alloc(alignment, size);
   CHECK(mapping && size >= bytes);
   memset(mapping, 0, size);
   PFN_vkGetMemoryHostPointerPropertiesEXT host_properties =
      (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(
         device, "vkGetMemoryHostPointerPropertiesEXT");
   CHECK(host_properties);
   VkMemoryHostPointerPropertiesEXT pointer = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
   };
   VK(host_properties(device, handle, mapping, &pointer));
   VkImportMemoryHostPointerInfoEXT import = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
      .handleType = handle, .pHostPointer = mapping,
   };
   VkMemoryAllocateFlagsInfo flags = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = &import,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   const VkMemoryPropertyFlags coherent =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   VkMemoryAllocateInfo allocate = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size,
      .pNext = usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT ? (void *)&flags : (void *)&import,
      .memoryTypeIndex = memory_type(lvp.physical, pointer.memoryTypeBits & requirements.memoryTypeBits, coherent),
   };
   VkDeviceMemory memory;
   VK(vkAllocateMemory(device, &allocate, NULL, &memory));
   VK(vkBindBufferMemory(device, buffer, memory, 0));
   return (struct imported_buffer){buffer, memory, mapping};
}

static void
close_lavapipe(struct lavapipe lvp, struct imported_buffer imported)
{
   vkDestroyBuffer(lvp.device, imported.buffer, NULL);
   vkFreeMemory(lvp.device, imported.memory, NULL);
   free(imported.mapping);
   vkDestroyDevice(lvp.device, NULL);
   vkDestroyInstance(lvp.instance, NULL);
}

static VkCommandBuffer
begin_commands(struct lavapipe lvp, VkCommandPool *pool)
{
   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = lvp.family,
   };
   VK(vkCreateCommandPool(lvp.device, &pool_info, NULL, pool));
   const VkCommandBufferAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = *pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   VK(vkAllocateCommandBuffers(lvp.device, &alloc, &cmd));
   const VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   VK(vkBeginCommandBuffer(cmd, &begin));
   return cmd;
}

static void
finish_commands(struct lavapipe lvp, VkCommandBuffer cmd,
                VkPipelineStageFlags2 stage, VkAccessFlags2 access)
{
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = stage, .srcAccessMask = access,
      .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
   };
   const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier,
   };
   vkCmdPipelineBarrier2(cmd, &dependency);
   VK(vkEndCommandBuffer(cmd));
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   VK(vkQueueSubmit(lvp.queue, 1, &submit, VK_NULL_HANDLE));
   VK(vkQueueWaitIdle(lvp.queue));
}

static void
check_compute(struct lavapipe lvp, struct imported_buffer imported, nir_shader *nir,
               const uint32_t cases[][4], unsigned case_count, const char *name)
{
   CHECK(case_count <= bytes / (table_stride + 16));
   VkDevice device = lvp.device;
   unsigned char *mapping = imported.mapping;
   VkBufferDeviceAddressInfo addr_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = imported.buffer,
   };
   const VkDeviceAddress address = vkGetBufferDeviceAddress(device, &addr_info);
   CHECK(address);
   for (unsigned i = 0; i < case_count; i++) {
      const size_t data_offset = case_count * table_stride + 16 * i;
      const uint64_t data = address + data_offset, length = 12;
      memcpy(mapping + i * table_stride + STELA_METAL_TABLE_BUFFER_ADDRESSES, &data, sizeof(data));
      memcpy(mapping + i * table_stride + STELA_METAL_TABLE_BUFFER_LENGTHS, &length, sizeof(length));
      memcpy(mapping + data_offset, cases[i], 12);
      memset(mapping + data_offset + 12, 0xa7, 4);
   }

   CHECK(nir->info.stage == MESA_SHADER_COMPUTE);
   VkPushConstantRange range = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = STELA_METAL_PUSH_BYTES };
   VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
      .pPushConstantRanges = &range,
   };
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));
   VkPipelineShaderStageNirCreateInfoMESA compiled = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA, .nir = nir,
   };
   VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .pName = "main", .pNext = &compiled },
   };
   VkPipeline pipeline;
   /* This fixture is compiled with generic options. Mesa clones the NIR
    * and sets its own options when the input has none. */
   const nir_shader_compiler_options *options = nir->options;
   nir->options = NULL;
   VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline));
   nir->options = options;
   VkCommandPool pool;
   VkCommandBuffer cmd = begin_commands(lvp, &pool);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   for (unsigned i = 0; i < case_count; i++) {
      uint64_t push[STELA_METAL_PUSH_BYTES / sizeof(uint64_t)] = { address + i * table_stride };
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
      vkCmdDispatch(cmd, 1, 1, 1);
   }
   finish_commands(lvp, cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
   for (unsigned i = 0; i < case_count; i++) {
      uint32_t words[4];
      memcpy(words, mapping + case_count * table_stride + 16 * i, sizeof(words));
      printf("%s case %u got=0x%08" PRIx32 " expected=0x%08" PRIx32 "\n",
             name, i, words[0], cases[i][3]);
      CHECK(words[0] == cases[i][3] && words[1] == cases[i][1] && words[2] == cases[i][2] &&
            words[3] == UINT32_C(0xa7a7a7a7));
   }
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   puts("AIR NIR -> lavapipe -> original imported allocation: PASS");
}

void
stela_test_lavapipe_software_half(nir_shader *nir)
{
   const struct lavapipe lvp = open_lavapipe();
   const struct imported_buffer imported = import_buffer(
      lvp, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
   check_compute(lvp, imported, nir, half_cases, ARRAY_SIZE(half_cases), "AIR FMA");
   close_lavapipe(lvp, imported);
}

void
stela_test_lavapipe_fp32_preserve(nir_shader *preserve, nir_shader *flush)
{
   static const uint32_t preserved[][4] = {
      {0x00400000, 0x00800000, 0, 0x4f},
      {0x80400000, 0x80800000, 0, 0x57},
      {0x3fc00000, 0x40400000, 0, 0x4f},
   };
   static const uint32_t flushed[][4] = {
      {0x00400000, 0x00800000, 0, 0x30},
      {0x80400000, 0x80800000, 0, 0x30},
      {0x3fc00000, 0x40400000, 0, 0x4f},
   };
   const struct lavapipe lvp = open_lavapipe();
   const struct imported_buffer imported = import_buffer(
      lvp, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
   check_compute(lvp, imported, preserve, preserved, ARRAY_SIZE(preserved), "AIR FP32 preserve");
   check_compute(lvp, imported, flush, flushed, ARRAY_SIZE(flushed), "AIR FP32 flush");
   check_compute(lvp, imported, preserve, preserved, ARRAY_SIZE(preserved), "AIR FP32 preserve again");
   close_lavapipe(lvp, imported);
}

static VkPipeline
render_pipeline(VkDevice device, VkPipelineLayout layout, VkFormat format,
                const VkRect2D *area, nir_shader *vertex, nir_shader *fragment)
{
   CHECK(vertex->info.stage == MESA_SHADER_VERTEX && fragment->info.stage == MESA_SHADER_FRAGMENT);
   const VkPipelineShaderStageNirCreateInfoMESA vertex_nir = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA, .nir = vertex,
   };
   const VkPipelineShaderStageNirCreateInfoMESA fragment_nir = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA, .nir = fragment,
   };
   const VkPipelineShaderStageCreateInfo stages[] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &vertex_nir,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &fragment_nir,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .pName = "main"},
   };
   const VkPipelineVertexInputStateCreateInfo inputs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {.width = area->extent.width, .height = area->extent.height, .maxDepth = 1};
   const VkPipelineViewportStateCreateInfo viewport_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = area,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendAttachmentState color_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo color = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &color_attachment,
   };
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &format,
   };
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering, .layout = layout,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &inputs, .pInputAssemblyState = &assembly,
      .pViewportState = &viewport_info, .pRasterizationState = &raster,
      .pMultisampleState = &multisample, .pColorBlendState = &color,
   };
   VkPipeline pipeline;
   const nir_shader_compiler_options *vertex_options = vertex->options;
   const nir_shader_compiler_options *fragment_options = fragment->options;
   vertex->options = NULL;
   fragment->options = NULL;
   VK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
   vertex->options = vertex_options;
   fragment->options = fragment_options;
   return pipeline;
}

/* Check the varying, half fragment arithmetic, rasterization, and copy to
 * the original host allocation. */
void
stela_test_lavapipe_software_render(nir_shader *vertex, nir_shader *fragment)
{
   const struct lavapipe lvp = open_lavapipe();
   const struct imported_buffer readback = import_buffer(lvp, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   VkDevice device = lvp.device;
   enum { width = 4, height = 4, pixel_bytes = 4, guard = 16 };
   const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkRect2D area = {.extent = {width, height}};
   const VkImageSubresourceRange subresources = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1,
   };
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
   };
   VkImage image;
   VK(vkCreateImage(device, &image_info, NULL, &image));
   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, image, &requirements);
   const VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(lvp.physical, requirements.memoryTypeBits, 0),
   };
   VkDeviceMemory memory;
   VK(vkAllocateMemory(device, &allocation, NULL, &memory));
   VK(vkBindImageMemory(device, image, memory, 0));
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format, .subresourceRange = subresources,
   };
   VkImageView view;
   VK(vkCreateImageView(device, &view_info, NULL, &view));
   const VkPipelineLayoutCreateInfo layout_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));
   VkPipeline pipeline = render_pipeline(device, layout, format, &area, vertex, fragment);

   memset(readback.mapping, 0xa7, 2 * guard + width * height * pixel_bytes);
   VkCommandPool pool;
   VkCommandBuffer cmd = begin_commands(lvp, &pool);
   VkImageMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image, .subresourceRange = subresources,
   };
   const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier,
   };
   vkCmdPipelineBarrier2(cmd, &dependency);
   const VkRenderingAttachmentInfo attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo render_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = area, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &attachment,
   };
   vkCmdBeginRendering(cmd, &render_info);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRendering(cmd);
   barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
   barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
   barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
   barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
   barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier2(cmd, &dependency);
   const VkBufferImageCopy copy = {
      .bufferOffset = guard, .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
      .imageExtent = {width, height, 1},
   };
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &copy);
   finish_commands(lvp, cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

   const unsigned char expected[] = {191, 64, 32, 255};
   const unsigned char *pixels = readback.mapping + guard;
   for (unsigned i = 0; i < width * height; i++) {
      const unsigned char *pixel = pixels + i * pixel_bytes;
      if (memcmp(pixel, expected, sizeof(expected)) != 0) {
         fprintf(stderr, "pixel %u got=%u,%u,%u,%u expected=%u,%u,%u,%u\n", i,
                 pixel[0], pixel[1], pixel[2], pixel[3], expected[0], expected[1], expected[2], expected[3]);
         exit(EXIT_FAILURE);
      }
   }
   for (unsigned i = 0; i < guard; i++) {
      CHECK(readback.mapping[i] == 0xa7);
      CHECK(pixels[width * height * pixel_bytes + i] == 0xa7);
   }
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyImageView(device, view, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, memory, NULL);
   close_lavapipe(lvp, readback);
   puts("AIR vertex/fragment NIR -> lavapipe -> 16 expected RGBA pixels: PASS");
}
