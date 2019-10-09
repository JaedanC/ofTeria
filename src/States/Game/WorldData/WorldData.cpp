#include "WorldData.h"
#include "Chunk.h"
#include "../WorldSpawn.h"

WorldData::WorldData(const string& worldName, WorldSpawn* worldSpawn)
	: worldName(worldName), worldFile(worldName), worldSpawn(worldSpawn)
{

}

inline ofVec2f WorldData::convertChunkIdToVec(int id)
{
	return ofVec2f(id % numChunksX, id / numChunksX);
}

inline int WorldData::convertChunkVecToId(const ofVec2f& vec)
{
	return (int)(vec.y * numChunksX + vec.x);
}

inline ofxMemoryMapping* WorldData::getWorldFile()
{
	return &worldFile;
}

inline size_t WorldData::getChunkDataSize()
{
	return sizeof(ChunkSaved) + chunkWidth * chunkHeight * sizeof(Block);
}

inline WorldSpawn* WorldData::getWorldSpawn()
{
	return worldSpawn;
}

inline ofVec2f WorldData::getBlockDim()
{
	return ofVec2f(blockWidth, blockHeight);
}

void WorldData::updateChunks()
{
	// TODO: Implement this function
	ofVec2f* playerPos = getWorldSpawn()->getEntityController()->getPlayer()->getWorldPos();
	ofVec2f copy = *(playerPos);

	copy.x = copy.x / blockWidth;
	copy.y = copy.y / blockHeight;

	// TODO: Incorporate with zoom.

	// TODO: This is not correct
	for (auto& chunkPair : loadedChunks) {
		freeChunk(chunkPair.second);
	}

	getBlock(copy);
}

void WorldData::freeChunk(Chunk* chunk)
{
	ofVec2f& chunkPos = chunk->getSaveDataPtr()->chunkPos;
	freeChunk(chunkPos);
}

void WorldData::freeChunk(const ofVec2f& chunkPos)
{
	if (loadedChunks.count(chunkPos) == 0) {
		cout << "WorldData::freeChunk: chunkDoes not exist. Can't free\n";
		return;
	}

	Chunk* chunk = loadedChunks[chunkPos];
	loadedChunks.erase(chunkPos);
	delete[] chunk;
}

Chunk* WorldData::loadChunk(const ofVec2f& chunkPos)
{
	int chunkId = convertChunkVecToId(chunkPos);
	int offset = chunkId * getChunkDataSize();

	Chunk* chunk = new Chunk(chunkPos, chunkWidth, chunkHeight, this);

	worldFile.read(chunk->getSaveDataPtr(), offset, getChunkDataSize());
	loadedChunks[chunkPos] = chunk;
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

	if (loadedChunks.count(chunkPos) != 0) {
		chunk = loadedChunks[chunkPos];
		return chunk;
	}

	return loadChunk(chunkPos);
}
