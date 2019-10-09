#pragma once
#ifndef WORLD_DATA_H
#define WORLD_DATA_H
#include "ofMain.h"
#include "Chunk.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"

class WorldData {
private:
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
	WorldData(const string& worldName);

	inline ofVec2f convertChunkIdToVec(int id) { return ofVec2f(id % numChunksX, id / numChunksX); }
	inline int convertChunkVecToId(const ofVec2f& vec) { return (int)(vec.y * numChunksX + vec.x); }
	inline ofxMemoryMapping* getWorldFile() { return &worldFile; }

	void updateChunks();
	void freeChunk(ofVec2f& chunkPos);
	void loadChunk(ofVec2f& chunkPos);

	Block* getBlock(ofVec2f worldPos);
	Block* getBlock(ofVec2f& chunkPos, ofVec2f& chunkRelativePos);
};


#endif /* WORLD_DATA_H */