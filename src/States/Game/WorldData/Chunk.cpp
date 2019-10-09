#include "Chunk.h"
#include "WorldData.h"
#include "ofxMemoryMapping/ofxMemoryMapping.h"

Chunk::Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, weak_ptr<WorldData> worldData)
	: save(chunkPos, chunkWidth, chunkHeight)
{
	blocks = new Block[chunkWidth * chunkHeight];
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

inline ChunkSaved* Chunk::getSaveDataPtr()
{
	return &save;
}
