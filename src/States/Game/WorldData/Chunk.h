#pragma once
#pragma once
#ifndef CHUNK_H
#define CHUNK_H
#include "ofMain.h"
#include "WorldData.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"

struct Block {
	short id;
	float debug;
	// Any fields here
};

struct ChunkSaved {
	ChunkSaved(ofVec2f chunkPos, int chunkWidth, int chunkHeight)
		: chunkPos(chunkPos), chunkWidth(chunkWidth), chunkHeight(chunkHeight)
	{
		numBlocks = chunkWidth * chunkHeight;
	}
	ofVec2f chunkPos;
	int chunkWidth;
	int chunkHeight;
	int numBlocks;
};

class Chunk {
private:
	WorldData* worldData;
	Block* blocks;
	ChunkSaved save;

public:
	Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData* worldData);
	void saveChunk();
	Block* getBlock(const ofVec2f& chunkRelativePos);
	Block* getBlock(int chunkIndex);
	inline ChunkSaved* getSaveDataPtr() { return &save; }
};


#endif /* CHUNK_H */