#pragma once
#ifndef WORLD_DATA_H
#define WORLD_DATA_H

#include "ofMain.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"
#include "Chunk.h"

class WorldSpawn;
class WorldData : public enable_shared_from_this<WorldData> {
private:
	string worldName;
	WorldSpawn * worldSpawn;
	shared_ptr<ofxMemoryMapping> worldFile;

	int blockWidth = 32;
	int blockHeight = 32;

	int chunkWidth = 4;
	int chunkHeight = 4;

	/*
	https://terraria.fandom.com/wiki/World_Size
	Large - 8400 blocks wide and 2400 blocks high, sky limit about 800-900 blocks
	above underground level. A Large world has 20,160,000 blocks, exactly 1.75x the 
	size of a Medium world and 4x the size of a Small world. */
	int worldWidth = 16;
	int worldHeight = 16;

	int numChunksX;
	int numChunksY;
	int numChunks;

	/* Stores a cache of the loaded chunks in a map. Retrievable by the chunkPos. */
	//            chunkId, Chunk* heap allocated
	unordered_map<int, Chunk*> loadedChunks;

public:
	WorldData(WorldSpawn* worldSpawn, const string& worldName);

	inline weak_ptr<ofxMemoryMapping> getWorldFile() { return worldFile; }
	inline WorldSpawn* getWorldSpawn() { return worldSpawn; }

	inline ofVec2f convertChunkIdToVec(int id) { return ofVec2f(id % numChunksX, id / numChunksX); }
	inline int convertChunkVecToId(const ofVec2f& vec) { return (int)(vec.y * numChunksX + vec.x); }
	inline ofVec2f getBlockDim() { return ofVec2f(blockWidth, blockHeight); }

	size_t getChunkDataSize();
	
	void temporaryCreateWorld();

	void updateChunks();
	void freeChunk(Chunk* chunk);
	void freeChunk(const ofVec2f& chunkPos);
	Chunk* loadChunk(const ofVec2f& chunkPos);

	Block* getBlock(const ofVec2f& worldPos);
	Block* getBlock(const ofVec2f& chunkPos, const ofVec2f& chunkRelativePos);
	Chunk* getChunk(const ofVec2f& chunkPos);
};


#endif /* WORLD_DATA_H */