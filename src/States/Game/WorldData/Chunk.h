#pragma once
#ifndef CHUNK_H
#define CHUNK_H

#include "ofMain.h"


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

class WorldData;
class Chunk {
private:
	WorldData* worldData;
	Block* blocks;
	ChunkSaved save;

public:
	Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData* worldData);
	inline ChunkSaved* getSaveDataPtr() { return &save; }
	inline WorldData* getWorldData() { return worldData; }

	void createRandomData();
	void freeData();

	void loadChunk(int chunkId);
	void saveChunk();
	Block* getBlock(const ofVec2f& chunkRelativePos);
	Block* getBlock(int chunkIndex);
};

#endif /* CHUNK_H */