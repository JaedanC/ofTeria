#include "Chunk.h"
#include "WorldData.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"

Chunk::Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData * worldData)
	: worldData(worldData->weak_from_this()), save(
		chunkPos, chunkWidth, chunkHeight
	)
{
}

void Chunk::saveChunk()
{
	auto worldFile = worldData.lock()->getWorldFile().lock();

	// This is the offset where to save in the file.
	int offset = worldData.lock()->convertChunkVecToId(save.chunkPos) * worldData.lock()->getChunkDataSize();

	// Save the Chunk MetaData first.
	worldFile->write(&save, offset, sizeof(ChunkSaved));
	offset += sizeof(ChunkSaved);

	// Next save the block heap data.
	worldFile->write(blocks, offset, save.numBlocks * sizeof(Block));
}

Block* Chunk::getBlock(const ofVec2f& chunkRelativePos)
{
	return getBlock(worldData.lock()->convertChunkVecToId(chunkRelativePos));
}

Block* Chunk::getBlock(int chunkIndex)
{
	return &blocks[chunkIndex];
}
