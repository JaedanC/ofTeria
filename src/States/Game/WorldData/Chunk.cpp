#include "Chunk.h"
#include "WorldData.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"

Chunk::Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData * worldData)
	: worldData(worldData), save(
		chunkPos, chunkWidth, chunkHeight
	)
{
	int blockWidth = getWorldData()->blockWidth;
	int blockHeight = getWorldData()->blockHeight;
	frameBuffer.allocate(save.chunkWidth * blockWidth, save.chunkHeight * blockHeight);
}

void Chunk::createRandomData()
{
	blocks = new Block[save.chunkWidth * save.chunkHeight];
	for (int i = 0; i < save.chunkWidth * save.chunkHeight; i++) {
		blocks[i].debug = i;
		blocks[i].id = i / 2;
	}
}

void Chunk::loadChunk(int chunkId)
{
	/* Reads the chunk from disk and loads the data into their corresponding buffers. */
	int offset = chunkId * getWorldData()->getChunkDataSize();
	getWorldData()->getWorldFile().lock()->read(getChunkMetaData(), offset, sizeof(ChunkSaved));
	offset += sizeof(ChunkSaved);
	blocks = new Block[save.chunkWidth * save.chunkHeight];
	getWorldData()->getWorldFile().lock()->read(blocks, offset, sizeof(Block) * save.numBlocks);

	/* When a chunk is loaded from memory a framebuffer for the chunk is immediately drawn. This saves
	GPU draws calls to per chunk not per block. Causes stuttering if too many chunks are loaded per frame. */
	frameBuffer.begin();
	int x, y;
	int blockWidth = getWorldData()->blockWidth;
	int blockHeight = getWorldData()->blockHeight;
	for (int i = 0; i < getChunkMetaData()->numBlocks; i++) {
		x = blockWidth * (i % getWorldData()->chunkWidth);
		y = blockHeight * (i / getWorldData()->chunkHeight);
		ofSetColor(abs(200 - (x / blockWidth) / 2 - (y / blockHeight) / 2) % 255 + 1, ((x / blockWidth) * 4 % 255), ((y / blockHeight) * 4 % 255));
		ofDrawRectangle(x, y, blockWidth, blockHeight);
	}
	frameBuffer.end();
}

void Chunk::saveChunk()
{
	auto worldFile = getWorldData()->getWorldFile().lock();

	// This is the offset where to save in the file.
	int chunkId = getWorldData()->convertChunkVecToId(save.chunkPos);
	int chunkSize = getWorldData()->getChunkDataSize();
	int offset = chunkId * chunkSize;

	// Save the Chunk MetaData first.
	worldFile->write(&save, offset, sizeof(ChunkSaved));
	offset += sizeof(ChunkSaved);

	// Next save the block heap data.
	worldFile->write(blocks, offset, save.numBlocks * sizeof(Block));
}

Block* Chunk::getBlock(const ofVec2f& chunkRelativePos)
{
	return getBlock(getWorldData()->convertChunkVecToId(chunkRelativePos));
}

Block* Chunk::getBlock(int blockIndex)
{
	return &blocks[blockIndex];
}
