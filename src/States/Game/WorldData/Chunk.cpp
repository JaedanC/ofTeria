#include "Chunk.h"
#include "WorldData.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"

Chunk::Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData * worldData)
	: worldData(worldData), save(
		chunkPos, chunkWidth, chunkHeight
	)
{
}

void Chunk::createRandomData()
{
	blocks = new Block[save.chunkWidth * save.chunkHeight];
	for (int i = 0; i < save.chunkWidth * save.chunkHeight; i++) {
		blocks[i].debug = i;
		blocks[i].id = i / 2;
	}
}

void Chunk::freeData()
{
	delete[] blocks;
}

void Chunk::loadChunk(int chunkId)
{
	int offset = chunkId * getWorldData()->getChunkDataSize();
	getWorldData()->getWorldFile().lock()->read(getChunkMetaData(), offset, sizeof(ChunkSaved));
	offset += sizeof(ChunkSaved);
	blocks = new Block[save.chunkWidth * save.chunkHeight];
	getWorldData()->getWorldFile().lock()->read(blocks, offset, sizeof(Block) * save.numBlocks);
}

void Chunk::saveChunk()
{
	auto worldFile = getWorldData()->getWorldFile().lock();

	// This is the offset where to save in the file.
	int chunkId = getWorldData()->convertChunkVecToId(save.chunkPos);
	int chunkSize = getWorldData()->getChunkDataSize();
	int offset = chunkId * chunkSize;

	/*cout << "Chunk::saveChunk() chunkId: " << chunkId << endl;
	cout << "Chunk::saveChunk() chunkSize: " << chunkSize << endl;
	cout << "Chunk::saveChunk() offset: " << offset << endl;*/

	// Save the Chunk MetaData first.
	//cout << "Chunk::saveChunk() Writing Chunk MetaData\n";
	worldFile->write(&save, offset, sizeof(ChunkSaved));
	offset += sizeof(ChunkSaved);

	// Next save the block heap data.
	//cout << "Chunk::saveChunk() Writing BlockData\n";
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
