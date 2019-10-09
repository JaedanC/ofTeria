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
	WorldData* worldData;
	Block* blocks;

	ChunkSaved save;
	Chunk(ofVec2f chunkPos, int chunkWidth, int chunkHeight, WorldData* worldData);

	void saveChunk();
	inline size_t sizeToSave();
};


#endif /* CHUNK_H */