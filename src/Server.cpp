#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include <iostream>
#include "Server.h"

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

Server::Server(std::string configPath)
    : gpus(),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(nullptr),
      activeRequestGpus(),
      currentCycle(0) {
    Config cfg = readConfig(configPath);

    // 1. Initialize GPUs
    for (const auto& gpuCfg : cfg.gpus) {
        std::size_t slots = gpuCfg.memoryBytes / cfg.blockSize;
        gpus.emplace_back(slots, cfg.blockSize);
    }

    // 2. Initialize Models
    for (const auto& modelCfg : cfg.models) {
        models.push_back(new Model(modelCfg.id, modelCfg.embeddingSize));
    }

    // 3. Initialize SchedulingPolicy
    if (cfg.policy == "arrivalOrder") {
        policy = new ArrivalOrderPolicy();
    } else if (cfg.policy == "shortestJob") {
        policy = new ShortestJobPolicy();
    } else if (cfg.policy == "longestPrefix") {
        policy = new LongestPrefixPolicy();
    } else {
        throw std::invalid_argument("unknown scheduling policy");
    }

    // 4. Initialize Requests as pending (nonSubmittedRequests)
    for (const auto& reqCfg : cfg.requests) {
        Model* model = findModelById(reqCfg.modelId);
        if (model == nullptr) {
            throw std::invalid_argument("unknown model id");
        }
        Request* req = nullptr;
        if (reqCfg.type == RequestType::Generative) {
            req = new GenerativeRequest(
                static_cast<int>(reqCfg.id),
                *model,
                reqCfg.admissionCycle,
                reqCfg.prompt,
                reqCfg.scriptedOutput,
                reqCfg.maxOutputTokens
            );
        } else {
            req = new EmbeddingRequest(
                static_cast<int>(reqCfg.id),
                *model,
                reqCfg.admissionCycle,
                reqCfg.prompt
            );
        }
        nonSubmittedRequests.push_back(req);
    }
}

Server::~Server() {
    clear();
}

Server::Server(const Server& other)
    : gpus(),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(nullptr),
      activeRequestGpus(),
      currentCycle(0) {
    copyFrom(other);
}

Server& Server::operator=(const Server& other) {
    if (this != &other) {
        clear();
        copyFrom(other);
    }
    return *this;
}

Server::Server(Server&& other) noexcept
    : gpus(std::move(other.gpus)),
      models(std::move(other.models)),
      nonSubmittedRequests(std::move(other.nonSubmittedRequests)),
      waitingRequests(std::move(other.waitingRequests)),
      activeRequests(std::move(other.activeRequests)),
      policy(other.policy),
      activeRequestGpus(std::move(other.activeRequestGpus)),
      currentCycle(other.currentCycle) {
    other.gpus.clear();
    other.models.clear();
    other.nonSubmittedRequests.clear();
    other.waitingRequests.clear();
    other.activeRequests.clear();
    other.activeRequestGpus.clear();
    other.policy = nullptr;
    other.currentCycle = 0;
}

Server& Server::operator=(Server&& other) noexcept {
    if (this != &other) {
        clear();

        gpus = std::move(other.gpus);
        models = std::move(other.models);
        nonSubmittedRequests = std::move(other.nonSubmittedRequests);
        waitingRequests = std::move(other.waitingRequests);
        activeRequests = std::move(other.activeRequests);
        policy = other.policy;
        activeRequestGpus = std::move(other.activeRequestGpus);
        currentCycle = other.currentCycle;

        other.gpus.clear();
        other.models.clear();
        other.nonSubmittedRequests.clear();
        other.waitingRequests.clear();
        other.activeRequests.clear();
        other.activeRequestGpus.clear();
        other.policy = nullptr;
        other.currentCycle = 0;
    }
    return *this;
}

void Server::clear() {
    for (Request* req : nonSubmittedRequests) {
        delete req;
    }
    nonSubmittedRequests.clear();

    for (Request* req : waitingRequests) {
        delete req;
    }
    waitingRequests.clear();

    for (Request* req : activeRequests) {
        delete req;
    }
    activeRequests.clear();
    activeRequestGpus.clear();

    for (Model* m : models) {
        delete m;
    }
    models.clear();

    delete policy;
    policy = nullptr;

    gpus.clear();
    currentCycle = 0;
}

void Server::copyFrom(const Server& other) {
    gpus = other.gpus;
    currentCycle = other.currentCycle;
    activeRequestGpus = other.activeRequestGpus;

    for (const Model* m : other.models) {
        models.push_back(new Model(*m));
    }

    if (other.policy != nullptr) {
        policy = other.policy->clone();
    }

    for (const Request* req : other.nonSubmittedRequests) {
        Model* copiedModel = findModelById(req->getModel().getId());
        nonSubmittedRequests.push_back(req->clone(*copiedModel));
    }

    for (const Request* req : other.waitingRequests) {
        Model* copiedModel = findModelById(req->getModel().getId());
        waitingRequests.push_back(req->clone(*copiedModel));
    }

    for (const Request* req : other.activeRequests) {
        Model* copiedModel = findModelById(req->getModel().getId());
        activeRequests.push_back(req->clone(*copiedModel));
    }
}

Model* Server::findModelById(const std::string& id) const {
    for (Model* m : models) {
        if (m->getId() == id) {
            return m;
        }
    }
    return nullptr;
}

bool Server::canEverFitOnAnyGPU(const Request* req) const {
    std::size_t maxMem = req->getMaxNeededMemory();
    for (const GPU& gpu : gpus) {
        std::size_t maxBlocks = gpu.getBlocksForMemory(maxMem);
        if (maxBlocks <= gpu.getBlocksNum()) {
            return true;
        }
    }
    return false;
}

int Server::selectBestGPUForRequest(const Request* req, std::vector<PrefixMatch>& matchesOut) const {
    matchesOut.clear();
    matchesOut.reserve(gpus.size());

    std::vector<std::string> cacheableTokens = req->getCacheableTokens();
    const Model& model = req->getModel();

    for (const GPU& gpu : gpus) {
        if (!cacheableTokens.empty()) {
            matchesOut.push_back(gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), cacheableTokens));
        } else {
            matchesOut.push_back(PrefixMatch{0, {}});
        }
    }

    int bestGpuIdx = -1;
    std::size_t bestCachedBlocks = 0;
    std::size_t bestTotalNeededBlocks = 0;

    for (std::size_t g = 0; g < gpus.size(); ++g) {
        const GPU& gpu = gpus[g];
        std::size_t newBlocks = gpu.getBlocksForMemory(req->getNeededMemory(gpu));

        std::size_t idleCachedBlocks = 0;
        for (std::size_t slot : matchesOut[g].blockIds) {
            if (gpu.isBlockIdle(slot)) {
                ++idleCachedBlocks;
            }
        }

        std::size_t requiredSlots = newBlocks + idleCachedBlocks;
        if (requiredSlots > gpu.availableSlots()) {
            continue;
        }

        std::size_t cachedBlocks = matchesOut[g].blockIds.size();

        if (bestGpuIdx == -1 ||
            cachedBlocks > bestCachedBlocks ||
            (cachedBlocks == bestCachedBlocks && requiredSlots < bestTotalNeededBlocks)) {
            bestGpuIdx = static_cast<int>(g);
            bestCachedBlocks = cachedBlocks;
            bestTotalNeededBlocks = requiredSlots;
        }
    }

    return bestGpuIdx;
}

bool Server::step() {
    if (nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty()) {
        return false;
    }

    std::vector<Request*> remainingNonSubmitted;
    for (Request* req : nonSubmittedRequests) {
        if (req->getAdmissionCycle() == currentCycle) {
            if (!canEverFitOnAnyGPU(req)) {
                std::cout << "rejected " << req->getId() << " cycle=" << currentCycle << "\n";
                delete req;
            } else {
                waitingRequests.push_back(req);
            }
        } else {
            remainingNonSubmitted.push_back(req);
        }
    }
    nonSubmittedRequests = std::move(remainingNonSubmitted);

    if (!waitingRequests.empty() && policy != nullptr) {
        std::vector<const Request*> constWaiting(waitingRequests.begin(), waitingRequests.end());
        std::vector<int> order = policy->choose(constWaiting, gpus);

        std::vector<bool> admitted(waitingRequests.size(), false);

        for (int idx : order) {
            if (idx < 0 || static_cast<std::size_t>(idx) >= waitingRequests.size()) {
                continue;
            }
            Request* req = waitingRequests[idx];
            std::vector<PrefixMatch> matches;
            int chosenGpuIdx = selectBestGPUForRequest(req, matches);

            if (chosenGpuIdx != -1) {
                GPU& chosenGpu = gpus[chosenGpuIdx];

                int bestSourceGpuIdx = -1;
                std::size_t longestTokens = matches[chosenGpuIdx].matchedTokens;

                for (std::size_t g = 0; g < gpus.size(); ++g) {
                    if (static_cast<int>(g) == chosenGpuIdx) {
                        continue;
                    }
                    if (matches[g].matchedTokens > longestTokens) {
                        longestTokens = matches[g].matchedTokens;
                        bestSourceGpuIdx = static_cast<int>(g);
                    }
                }

                if (bestSourceGpuIdx != -1) {
                    chosenGpu.copyMemoryFromOtherGPU(gpus[bestSourceGpuIdx], matches[bestSourceGpuIdx].blockIds);
                }

                req->reserveMemory(chosenGpu);
                activeRequests.push_back(req);
                activeRequestGpus.push_back(static_cast<std::size_t>(chosenGpuIdx));
                admitted[idx] = true;
            }
        }

        std::vector<Request*> remainingWaiting;
        for (std::size_t i = 0; i < waitingRequests.size(); ++i) {
            if (!admitted[i]) {
                remainingWaiting.push_back(waitingRequests[i]);
            }
        }
        waitingRequests = std::move(remainingWaiting);
    }

    std::vector<Request*> remainingActive;
    std::vector<std::size_t> remainingActiveGpus;

    for (std::size_t i = 0; i < activeRequests.size(); ++i) {
        Request* req = activeRequests[i];
        std::size_t gpuIdx = activeRequestGpus[i];
        GPU& gpu = gpus[gpuIdx];

        req->act(gpu);

        if (req->finished()) {
            std::cout << "finished " << req->getId()
                      << " cycle=" << currentCycle
                      << " gpu=" << gpuIdx << ": "
                      << req->getOutput() << "\n";
            req->deallocateMemory(gpu);
            delete req;
        } else {
            remainingActive.push_back(req);
            remainingActiveGpus.push_back(gpuIdx);
        }
    }

    activeRequests = std::move(remainingActive);
    activeRequestGpus = std::move(remainingActiveGpus);

    ++currentCycle;
    return true;
}

void Server::serve() {
    while (step()) {
    }

    std::size_t totalCacheHits = 0;
    std::size_t totalEvictions = 0;
    for (const GPU& gpu : gpus) {
        totalCacheHits += gpu.getCacheHits();
        totalEvictions += gpu.getEvictions();
    }

    std::cout << "summary: cycles=" << currentCycle
              << " cache_hits=" << totalCacheHits
              << " evictions=" << totalEvictions << "\n";
}