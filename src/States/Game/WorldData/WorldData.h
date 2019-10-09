#pragma once
#ifndef WORLD_DATA_H
#define WORLD_DATA_H
#include "ofMain.h"
#include "Chunk.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"

class WorldSpawn;
class WorldData {
private:
	WorldSpawn* worldSpawn;

	int blockWidth = 32;
	int blockHeight = 32;

	int chunkWidth = 128;
	int chunkHeight = 128;

	/*
	https://terraria.fandom.com/wiki/World_Size
	Large - 8400 blocks wide and 2400 blocks high, sky limit about 800-900 blocks
	above underground level. A Large world has 20,160,000 blocks, exactly 1.75x the 
	size of a Medium world and 4x the size of a Small world. */
	int worldWidth = 8400;
	int worldHeight = 2400;

	int numChunksX = ceil((float)worldWidth / chunkWidth);
	int numChunksY = ceil((float)worldHeight / chunkHeight);
	int numChunks = numChunks * numChunksY;

	string worldName;
	ofxMemoryMapping worldFile;

	/* Stores a cache of the loaded chunks in a map. Retrievable by the chunkPos. */
	unordered_map<ofVec2f, Chunk*> loadedChunks;

public:
	WorldData(const string& worldName, WorldSpawn* worldSpawn);

	inline ofVec2f convertChunkIdToVec(int id);
	inline int convertChunkVecToId(const ofVec2f& vec);
	inline ofxMemoryMapping* getWorldFile();
	inline size_t getChunkDataSize();
	inline WorldSpawn* getWorldSpawn();
	inline ofVec2f getBlockDim();

	void updateChunks();
	void freeChunk(Chunk* chunk);
	void freeChunk(const ofVec2f& chunkPos);
	Chunk* loadChunk(const ofVec2f& chunkPos);

	Block* getBlock(const ofVec2f& worldPos);
	Block* getBlock(const ofVec2f& chunkPos, const ofVec2f& chunkRelativePos);
	Chunk* getChunk(const ofVec2f& chunkPos);
};


#endif /* WORLD_DATA_H */