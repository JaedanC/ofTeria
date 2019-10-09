#include "Chunk.h"
#include "WorldData.h"

Chunk::Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData* worldData)
	: save(chunkPos, chunkWidth, chunkHeight)
{
	blocks = new Block[chunkWidth * chunkHeight];
}

void Chunk::saveChunk()
{
	ofxMemoryMapping* worldFile = worldData->getWorldFile();

	// This is the offset where to save in the file.
	int offset = worldData->convertChunkVecToId(save.chunkPos) * worldData->getChunkDataSize();
	
	// Save the Chunk MetaData first.
	worldFile->write(&save, offset, sizeof(ChunkSaved));
	offset += sizeof(ChunkSaved);

	// Next save the block heap data.
	worldFile->write(blocks, offset, save.numBlocks * sizeof(Block));
}

Block* Chunk::getBlock(const ofVec2f& chunkRelativePos)
{
	return getBlock(worldData->convertChunkVecToId(chunkRelativePos));
}

Block* Chunk::getBlock(int chunkIndex)
{
	return &blocks[chunkIndex];
}

inline ChunkSaved* Chunk::getSaveDataPtr()
{
	return &save;
}
