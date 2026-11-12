// SPDX-License-Identifier: GPL-2.0-or-later
// Hosted CPU Vulkan probe. Select the lavapipe ICD with VK_DRIVER_FILES.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

#define VK_CALL(expression)                                                                                  \
    do {                                                                                                     \
        VkResult result = (expression);                                                                      \
        if (result != VK_SUCCESS) {                                                                          \
            fprintf(stderr, "%s returned VkResult %d\n", #expression, result);                               \
            exit(EXIT_FAILURE);                                                                              \
        }                                                                                                    \
    } while (0)

static _Noreturn void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static VkPhysicalDevice find_cpu(VkInstance instance, VkDriverId expected_driver) {
    uint32_t count = 0;
    VK_CALL(vkEnumeratePhysicalDevices(instance, &count, NULL));
    if (count == 0)
        fail("No Vulkan devices");
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (!devices)
        fail("Could not allocate the Vulkan device list");
    VK_CALL(vkEnumeratePhysicalDevices(instance, &count, devices));
    VkPhysicalDevice device = VK_NULL_HANDLE;
    for (uint32_t index = 0; index < count; ++index) {
        VkPhysicalDeviceDriverProperties driver = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
        };
        VkPhysicalDeviceProperties2 properties = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &driver,
        };
        vkGetPhysicalDeviceProperties2(devices[index], &properties);
        if (properties.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU ||
            driver.driverID != expected_driver)
            continue;
        printf("device=%s driver=%s info=%s vulkan=%u.%u.%u\n", properties.properties.deviceName,
               driver.driverName, driver.driverInfo, VK_API_VERSION_MAJOR(properties.properties.apiVersion),
               VK_API_VERSION_MINOR(properties.properties.apiVersion),
               VK_API_VERSION_PATCH(properties.properties.apiVersion));
        device = devices[index];
        break;
    }
    free(devices);
    if (!device)
        fail("Requested CPU Vulkan device was not found");
    return device;
}

static uint32_t compute_queue(VkPhysicalDevice physical) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    VkQueueFamilyProperties *queues = calloc(count, sizeof(*queues));
    if (!queues)
        fail("Could not allocate the Vulkan queue list");
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
    uint32_t family = 0;
    while (family < count && !(queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
        ++family;
    free(queues);
    if (family == count)
        fail("The CPU device has no compute queue");
    return family;
}

static uint32_t coherent_type(VkPhysicalDevice physical, uint32_t mask) {
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
        if ((mask & (1u << index)) && (memory.memoryTypes[index].propertyFlags & flags) == flags)
            return index;
    }
    fail("The storage buffer has no host-coherent memory type");
}

static VkShaderModule shader_module(VkDevice device, const char *path) {
    FILE *input = fopen(path, "rb");
    if (!input || fseek(input, 0, SEEK_END) != 0)
        fail("Could not open the SPIR-V shader");
    long size = ftell(input);
    if (size <= 0 || size > 64 * 1024 || size % 4 || fseek(input, 0, SEEK_SET) != 0)
        fail("Invalid SPIR-V shader size");
    uint32_t *words = malloc((size_t)size);
    if (!words || fread(words, 1, (size_t)size, input) != (size_t)size)
        fail("Could not read SPIR-V shader");
    fclose(input);
    VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)size,
        .pCode = words,
    };
    VkShaderModule module;
    VK_CALL(vkCreateShaderModule(device, &info, NULL, &module));
    free(words);
    return module;
}

struct ExtensionList {
    uint32_t count;
    VkExtensionProperties *items;
};

static struct ExtensionList extensions_of(VkPhysicalDevice physical) {
    struct ExtensionList list = {0};
    VK_CALL(vkEnumerateDeviceExtensionProperties(physical, NULL, &list.count, NULL));
    if (list.count != 0) {
        list.items = calloc(list.count, sizeof(*list.items));
        if (!list.items)
            fail("Could not allocate the Vulkan extension list");
        VK_CALL(vkEnumerateDeviceExtensionProperties(physical, NULL, &list.count, list.items));
    }
    return list;
}

static int has_extension(struct ExtensionList list, const char *name) {
    for (uint32_t index = 0; index < list.count; ++index) {
        if (strcmp(list.items[index].extensionName, name) == 0)
            return 1;
    }
    return 0;
}

static void report_features(VkPhysicalDevice physical) {
    struct ExtensionList extensions = extensions_of(physical);
    VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT unused = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT,
    };
    VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
    };
    VkPhysicalDeviceRobustness2FeaturesEXT robustness2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
    };
    VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
    };
    void *next = NULL;
#define QUERY(feature, extension)                                                                            \
    do {                                                                                                     \
        if (has_extension(extensions, extension)) {                                                          \
            feature.pNext = next;                                                                            \
            next = &feature;                                                                                 \
        }                                                                                                    \
    } while (0)
    QUERY(unused, VK_EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_EXTENSION_NAME);
    QUERY(conditional, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
    QUERY(robustness2, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    QUERY(maintenance5, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
#ifdef VK_KHR_unified_image_layouts
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR,
    };
    QUERY(unified, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
#endif
#undef QUERY
    VkPhysicalDeviceVulkan13Features vulkan13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = next,
    };
    VkPhysicalDeviceVulkan12Features vulkan12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &vulkan13,
    };
    VkPhysicalDeviceFeatures2 features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &vulkan12,
    };
    vkGetPhysicalDeviceFeatures2(physical, &features);
    printf("bufferDeviceAddress=%u timelineSemaphore=%u synchronization2=%u shaderFloat16=%u "
           "dynamicRendering=%u\n",
           vulkan12.bufferDeviceAddress, vulkan12.timelineSemaphore, vulkan13.synchronization2,
           vulkan12.shaderFloat16, vulkan13.dynamicRendering);
    printf("descriptorIndexing=%u runtimeDescriptorArray=%u descriptorBindingPartiallyBound=%u\n",
           vulkan12.descriptorIndexing, vulkan12.runtimeDescriptorArray,
           vulkan12.descriptorBindingPartiallyBound);
    printf("descriptorBindingUpdateAfterBind=%d shaderImageArrayNonUniformIndexing=%d\n",
           vulkan12.descriptorBindingSampledImageUpdateAfterBind &&
               vulkan12.descriptorBindingStorageImageUpdateAfterBind &&
               vulkan12.descriptorBindingUniformTexelBufferUpdateAfterBind &&
               vulkan12.descriptorBindingStorageTexelBufferUpdateAfterBind &&
               vulkan12.descriptorBindingUpdateUnusedWhilePending,
           vulkan12.shaderSampledImageArrayNonUniformIndexing &&
               vulkan12.shaderStorageImageArrayNonUniformIndexing);
    printf("vulkanMemoryModel=%u storageImageWithoutFormat=%d independentBlend=%u hostQueryReset=%u\n",
           vulkan12.vulkanMemoryModel,
           features.features.shaderStorageImageReadWithoutFormat &&
               features.features.shaderStorageImageWriteWithoutFormat,
           features.features.independentBlend, vulkan12.hostQueryReset);
    VkPhysicalDeviceExtendedDynamicState3PropertiesEXT state3 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 properties = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    };
    if (has_extension(extensions, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME))
        properties.pNext = &state3;
    vkGetPhysicalDeviceProperties2(physical, &properties);
    printf("extendedDynamicState=%d maintenance5=%d nullDescriptor=%u conditionalRendering=%u\n",
           properties.properties.apiVersion >= VK_API_VERSION_1_3,
           vulkan13.maintenance4 && maintenance5.maintenance5, robustness2.nullDescriptor,
           conditional.conditionalRendering);
    printf("dynamicRenderingUnusedAttachments=%u dynamicPrimitiveTopologyUnrestricted=%u\n",
           unused.dynamicRenderingUnusedAttachments, state3.dynamicPrimitiveTopologyUnrestricted);
    printf("pciBusInfo=%d dmaBufImport=%d externalMemoryHost=%d\n",
           has_extension(extensions, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME),
           has_extension(extensions, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) &&
               has_extension(extensions, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME),
           has_extension(extensions, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME));
    if (!has_extension(extensions, "VK_KHR_unified_image_layouts")) {
        puts("unifiedImageLayouts=0");
    } else {
#ifdef VK_KHR_unified_image_layouts
        printf("unifiedImageLayouts=%u\n", unified.unifiedImageLayouts);
#else
        puts("unifiedImageLayouts=unknown (extension present, headers lack feature query)");
#endif
    }
    free(extensions.items);
}

static void compute(VkDevice device, uint32_t family, VkPhysicalDevice physical, const char *shader) {
    enum { groups = 4, values = groups * 16 };
    const VkDeviceSize bytes = values * sizeof(uint32_t);
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);

    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    };
    VkBuffer buffer;
    VK_CALL(vkCreateBuffer(device, &buffer_info, NULL, &buffer));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkMemoryAllocateInfo allocation = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = coherent_type(physical, requirements.memoryTypeBits),
    };
    VkDeviceMemory memory;
    VK_CALL(vkAllocateMemory(device, &allocation, NULL, &memory));
    VK_CALL(vkBindBufferMemory(device, buffer, memory, 0));
    void *mapped;
    VK_CALL(vkMapMemory(device, memory, 0, bytes, 0, &mapped));
    memset(mapped, 0, (size_t)bytes);

    VkDescriptorSetLayoutBinding binding = {
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VkDescriptorSetLayout set_layout;
    VK_CALL(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL, &set_layout));
    VkDescriptorPoolSize pool_size = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1};
    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorPool descriptor_pool;
    VK_CALL(vkCreateDescriptorPool(device, &descriptor_pool_info, NULL, &descriptor_pool));
    VkDescriptorSetAllocateInfo set_allocation = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
    };
    VkDescriptorSet set;
    VK_CALL(vkAllocateDescriptorSets(device, &set_allocation, &set));
    VkDescriptorBufferInfo storage = {.buffer = buffer, .range = bytes};
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &storage,
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
    };
    VkPipelineLayout layout;
    VK_CALL(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL, &layout));
    VkShaderModule module = shader_module(device, shader);
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = layout,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = module,
                  .pName = "main"},
    };
    VkPipeline pipeline;
    VK_CALL(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline));

    VkCommandPoolCreateInfo command_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = family,
    };
    VkCommandPool command_pool;
    VK_CALL(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool));
    VkCommandBufferAllocateInfo command_allocation = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .commandBufferCount = 1,
    };
    VkCommandBuffer command;
    VK_CALL(vkAllocateCommandBuffers(device, &command_allocation, &command));
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CALL(vkBeginCommandBuffer(command, &begin));
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(command, groups, 1, 1);
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &barrier, 0, NULL, 0, NULL);
    VK_CALL(vkEndCommandBuffer(command));
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command};
    VK_CALL(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK_CALL(vkQueueWaitIdle(queue));

    const uint32_t *output = mapped;
    uint32_t checksum = 0;
    for (uint32_t index = 0; index < values; ++index) {
        if (output[index] != index * index + 17u) {
            fprintf(stderr, "Unexpected shader output at %u: %u\n", index, output[index]);
            exit(EXIT_FAILURE);
        }
        checksum += output[index];
    }
    printf("compute=pass values=%u checksum=%u\n", (unsigned)values, checksum);

    vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, module, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyDescriptorPool(device, descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(device, set_layout, NULL);
    vkUnmapMemory(device, memory);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && strcmp(argv[2], "--swiftshader") != 0))
        fail("usage: northbridge-smoke shader.spv [--swiftshader]");
    VkDriverId driver = argc == 3 ? VK_DRIVER_ID_GOOGLE_SWIFTSHADER : VK_DRIVER_ID_MESA_LLVMPIPE;
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Northbridge hosted smoke",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                          .pApplicationInfo = &app};
    VkInstance instance;
    VK_CALL(vkCreateInstance(&instance_info, NULL, &instance));
    VkPhysicalDevice physical = find_cpu(instance, driver);
    report_features(physical);

    uint32_t family = compute_queue(physical);
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                      .queueCreateInfoCount = 1,
                                      .pQueueCreateInfos = &queue};
    VkDevice device;
    VK_CALL(vkCreateDevice(physical, &device_info, NULL, &device));
    compute(device, family, physical, argv[1]);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return EXIT_SUCCESS;
}
