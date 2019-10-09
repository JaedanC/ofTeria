#include "Chunk.h"

Chunk::Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData* worldData)
	: save(chunkPos, chunkWidth, chunkHeight)
{
	blocks = new Block[chunkWidth * chunkHeight];
}

void Chunk::saveChunk()
{
	ofxMemoryMapping* worldFile = worldData->getWorldFile();

	// This is the offset where to save in the file.
	int offset = worldData->convertChunkVecToId(save.chunkPos) * sizeToSave();
	
	// Save the Chunk MetaData first.
	worldFile->write(&save, offset, sizeof(ChunkSaved));
	offset += sizeof(ChunkSaved);

	// Next save the block heap data.
	worldFile->write(blocks, offset, save.numBlocks * sizeof(Block));
}

size_t Chunk::sizeToSave()
{
	return sizeof(save) + save.numBlocks * sizeof(Block);
}
