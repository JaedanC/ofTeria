#pragma once
#ifndef CHUNK_H
#define CHUNK_H

#include "ofMain.h"

/* Any data that needs to be stored with a block should go here. Try to be stingy with dataSizes. 
No pointers allowed here. Everything must be by value. */
struct Block {
	short id;
	float debug;
	// Any fields here
};

/* Any data that need to be stored alongside a Chunk should be stored in here. Update the contructor, accordingly
if you change the fields. No pointers allowed in here. Everything should be by value. */
struct ChunkSaved {
	ChunkSaved(glm::uvec2 chunkPos, int chunkWidth, int chunkHeight)
		: chunkPos(chunkPos), chunkWidth(chunkWidth), chunkHeight(chunkHeight)
	{
		numBlocks = chunkWidth * chunkHeight;
	}
	glm::uvec2 chunkPos;
	int chunkWidth;
	int chunkHeight;
	int numBlocks;
};

/* This Class stores the Blocks and the Chunk meta data together. Use this class to query specific blocks
or load/save the data from disk. Blocks is not guaranteed to be defined if freeData or loadData have/haven't been
called. */
class WorldData;
class Chunk {
private:
	/* Pointer to the WorldData instance that owns us. */
	WorldData* worldData;
	Block* blocks;
	ChunkSaved save;

	bool fboLoaded = false;
public:
	ofFbo frameBuffer;

	/* Takes in a chunkPos, chunkWidth, chunkHeight and a pointer to the WorldData instance that owns us. */
	Chunk(glm::uvec2 chunkPos, int chunkWidth, int chunkHeight, WorldData* worldData);
	~Chunk();

	/* Returns a pointer to the WorldData instance that owns us. */
	inline WorldData* getWorldData() { return worldData; }

	/* Returns a pointer to the chunk Metadata struct. */
	inline ChunkSaved* getChunkMetaData() { return &save; }

	/* Allocate blocks on the heap and fills it with data. It's not actually random, but it's not useful data
	either. Just for testing purposes. */
	void createRandomData();

	void drawChunk(float worldX, float worldY);

	/* Saves this chunkBack to memory. */
	void saveChunk();


	/* Returns a pointer to the Block at the chunkRelativePosition of the block relative to the chunkPos in
	the world. */
	Block* getBlock(const glm::uvec2& chunkRelativePos);
	Block* getBlock(int blockIndex);

};

#endif /* CHUNK_H */