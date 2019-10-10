#include "WorldData.h"
#include "../WorldSpawn.h"

WorldData::WorldData(WorldSpawn* worldSpawn, const string& worldName)
	: worldSpawn(worldSpawn->weak_from_this())
{
	cout << "Making WorldData\n";
}

inline ofVec2f WorldData::convertChunkIdToVec(int id)
{
	return ofVec2f();
}

inline int WorldData::convertChunkVecToId(const ofVec2f& vec)
{
	return 0;
}

inline size_t WorldData::getChunkDataSize()
{
	return size_t();
}

inline ofVec2f WorldData::getBlockDim()
{
	return ofVec2f();
}

void WorldData::updateChunks()
{
}

void WorldData::freeChunk(Chunk* chunk)
{
}

void WorldData::freeChunk(const ofVec2f& chunkPos)
{
}

Chunk* WorldData::loadChunk(const ofVec2f& chunkPos)
{
	return nullptr;
}

Block* WorldData::getBlock(const ofVec2f& worldPos)
{
	return nullptr;
}

Block* WorldData::getBlock(const ofVec2f& chunkPos, const ofVec2f& chunkRelativePos)
{
	return nullptr;
}

Chunk* WorldData::getChunk(const ofVec2f& chunkPos)
{
	return nullptr;
}
