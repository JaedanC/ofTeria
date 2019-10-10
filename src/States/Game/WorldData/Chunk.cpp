#include "Chunk.h"
#include "WorldData.h"

Chunk::Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData * worldData)
	: worldData(worldData->weak_from_this()), save(
		chunkPos, chunkWidth, chunkHeight
	)
{
}

void Chunk::saveChunk()
{
}

Block* Chunk::getBlock(const ofVec2f& chunkRelativePos)
{
	return nullptr;
}

Block* Chunk::getBlock(int chunkIndex)
{
	return nullptr;
}
