#include "WorldData.h"
#include "Chunk.h"
#include "../WorldSpawn.h"

WorldData::WorldData(const string& worldName, WorldSpawn * worldSpawn)
	: worldName(worldName), worldFile(worldName), worldSpawn(worldSpawn)
{

}

void WorldData::updateChunks()
{
	// TODO: Implement this function
	ofVec2f * playerPos = getWorldSpawn()->getEntityController()->getPlayer()->getWorldPos();
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
	ofVec2f &chunkPos = chunk->getSaveDataPtr()->chunkPos;
	freeChunk(chunkPos);
}

void WorldData::freeChunk(const ofVec2f& chunkPos)
{
	if (loadedChunks.count(chunkPos) == 0) {
		cout << "WorldData::freeChunk: chunkDoes not exist. Can't free\n";
		return;
	}

	delete[] loadedChunks[chunkPos];
	loadedChunks.erase(chunkPos);
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

	ofVec2f chunkRelativePos = {
		(int)worldPos.x % chunkWidth,
		(int)worldPos.y % chunkHeight
	};

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
