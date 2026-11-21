#include "GPU.h"

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr) {}

std::size_t GPU::availableSlots() const {
    std::size_t count = 0;
    for (const auto& block : memory) {
        if (block == nullptr) {
            ++count;
        }
    }
    return count;
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    return (memory + blockSize - 1) / blockSize;
}

void GPU::reserve(std::size_t n, const Request& request) {
    // Implementation for reserving memory blocks
}

void GPU::unreserve(std::size_t n, const Request& request) {
    // Implementation for unreserving memory blocks
}

std::size_t GPU::allocateBlock(const Request& request) {
    // Implementation for allocating a memory block
    return 0; // Placeholder return value
}

void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    // Implementation for adding a memory block for GenerativeRequest
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot) {
    // Implementation for adding a memory block for EmbeddingRequest
}

void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId) {
    // Implementation for assigning prefix to a memory block
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const {
    // Implementation for finding the longest cached prefix
    return {0, {}}; // Placeholder return value
}

void copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks) {
    // Implementation for copying memory from another GPU
}

void GPU::acquire(std::size_t slot) {
    if (slot < memory.size() && memory[slot] != nullptr) {
        memory[slot]->acquire();
    }
}

void GPU::release(std::size_t slot) {
    if (slot < memory.size() && memory[slot] != nullptr) {
        IdleAction action = memory[slot]->release(); //add behaviour based on IdleAction returned
    }
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    if (blockId < memory.size() && memory[blockId] != nullptr) {
        const MemoryBlock* block = memory[blockId];
        return block->readFromBlock(offset);
    }
    return NULL;
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    if (blockId < memory.size() && memory[blockId] != nullptr) {
        MemoryBlock* block = memory[blockId];
        block->writeToBlock(offset, value);
    }
}