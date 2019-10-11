#include "WorldData.h"
#include "ofMain.h"
#include "ofVec2f.h"
#include "Chunk.h"
#include "../WorldSpawn.h"
#include "../Entities/EntityController.h"
#include "../Entities/Entity/Entity.h"
#include "../Entities/Entity/Player.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"
#include "../addons/ofxDebugger/ofxDebugger.h"


WorldData::WorldData(WorldSpawn* worldSpawn, const string& worldName)
	: worldSpawn(worldSpawn), worldFile(make_shared<ofxMemoryMapping>("worldsave.wld"))
{
	cout << "Constructing WorldData\n";

	numChunksX = ceil((float)worldWidth / chunkWidth);
	numChunksY = ceil((float)worldHeight / chunkHeight);
	numChunks = numChunksX * numChunksY;

	temporaryCreateWorld();
}

size_t WorldData::getChunkDataSize()
{
	return sizeof(ChunkSaved) + chunkWidth * chunkHeight * sizeof(Block);
}

void WorldData::temporaryCreateWorld()
{
	//num_chunks, numChunksX, numChunksY
	//chunkWidth, chunkHeight
	int dataSize = getChunkDataSize() * numChunks;
	getWorldFile().lock()->resize(dataSize);

	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		ofVec2f chunkPos = convertChunkIdToVec(chunkId);
		Chunk chunk(chunkPos, chunkWidth, chunkHeight, this);
		chunk.createRandomData();
		chunk.saveChunk();
		chunk.freeData();
	}
}

void WorldData::updateChunks()
{
	debugPush("WorldData::updateChunks()");
	// TODO: Implement this function
	ofVec2f* playerPos = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getWorldPos();
	ofVec2f copy = *(playerPos);

	copy.x = copy.x / blockWidth;
	copy.y = copy.y / blockHeight;

	// TODO: Incorporate with zoom.

	// TODO: This is not correct
	vector<int> toDelete;
	for (auto& chunkPair : loadedChunks) {
		freeChunk(chunkPair.second);
		toDelete.push_back(chunkPair.first);
	}

	for (auto& chunkId : toDelete) {
		loadedChunks.erase(chunkId);
	}

	auto* block = getBlock(copy);
}

void WorldData::freeChunk(Chunk* chunk)
{
	ofVec2f& chunkPos = chunk->getSaveDataPtr()->chunkPos;
	freeChunk(chunkPos);
}

void WorldData::freeChunk(const ofVec2f& chunkPos)
{
	int chunkId = convertChunkVecToId(chunkPos);
	if (loadedChunks.count(chunkId) == 0) {
		cout << "WorldData::freeChunk: chunkDoes not exist. Can't free\n";
		return;
	}

	Chunk* chunk = loadedChunks[chunkId];
	chunk->freeData();
	delete[] chunk;
}

Chunk* WorldData::loadChunk(const ofVec2f& chunkPos)
{
	int chunkId = convertChunkVecToId(chunkPos);
	int dataSize = getChunkDataSize();
	int offset = chunkId * dataSize;

	Chunk* chunk = new Chunk(chunkPos, chunkWidth, chunkHeight, this);
	chunk->loadChunk(chunkId);

	loadedChunks[chunkId] = chunk;
	return chunk;
}

Block* WorldData::getBlock(const ofVec2f& worldPos)
{
	ofVec2f chunkPos = {
		worldPos.x / chunkWidth,
		worldPos.y / chunkHeight
	};

	int a = static_cast<int>(worldPos.x) % chunkWidth;
	int b = static_cast<int>(worldPos.y) % chunkHeight;

	ofVec2f chunkRelativePos = { (float)a, (float)b };

	return getBlock(chunkPos, chunkRelativePos);
}

Block* WorldData::getBlock(const ofVec2f& chunkPos, const ofVec2f& chunkRelativePos)
{
	Chunk* chunk = getChunk(chunkPos);
	return chunk->getBlock(chunkRelativePos);
}

Chunk* WorldData::getChunk(const ofVec2f& chunkPos)
{
	Chunk* chunk;
	int chunkId = convertChunkVecToId(chunkPos);

	if (loadedChunks.count(chunkId) != 0) {
		chunk = loadedChunks[chunkId];
		return chunk;
	}

	return loadChunk(chunkPos);
}
