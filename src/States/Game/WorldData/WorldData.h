#pragma once
#ifndef WORLD_DATA_H
#define WORLD_DATA_H

#include "ofMain.h"
#include "../addons/ofxMemoryMapping/ofxMemoryMapping.h"
#include "Chunk.h"

class WorldSpawn;
class WorldData : public enable_shared_from_this<WorldData> {
private:
	/* Pointer to the WorldSpawn instance that owns us. */
	WorldSpawn * worldSpawn;

	/* Is the file that we are writing the world to. */
	shared_ptr<ofxMemoryMapping> worldFile;

	/* Stores a cache of the loaded chunks on the screen. Each of these chunks are checked to see if they should still
	exist or that more should be added every frame. Then these chunks are all drawn to the screen in draw(). */
	unordered_map<int, Chunk*> loadedChunks;

	/* These two are used to stop stuttering from loaded too many chunks from disk in the same frame. */
	int maxRenderedChunksPerFrame = 1;
	int renderedChunksInThisFrame = 0;
public:

	/* https://terraria.fandom.com/wiki/World_Size
	Large - 8400 blocks wide and 2400 blocks high, sky limit about 800-900 blocks
	above underground level. A Large world has 20,160,000 blocks, exactly 1.75x the 
	size of a Medium world and 4x the size of a Small world. */
	unsigned int worldWidth = 1024; // In blocks
	unsigned int worldHeight = 1024;
	unsigned int chunkWidth = 64; // In blocks
	unsigned int chunkHeight = 64;
	unsigned int blockWidth = 16; // In pixels @ zoom == 1
	unsigned int blockHeight = 16;

	/* These are initialised in the constructor. */
	unsigned int numChunksX;
	unsigned int numChunksY;
	unsigned int numChunks;

	/* Takes in a pointer the WorldSpawn instance that owns us. */
	WorldData(WorldSpawn* worldSpawn);
	~WorldData() {
		cout << "Destroying WorldData\n";
	};

	/* Draws the chunks that are loaded onto the screen. */
	void draw();

	/* Returns a weak pointer to the ofxMemoryMapping instance that we own. */
	inline weak_ptr<ofxMemoryMapping> getWorldFile() { return worldFile; }

	/* Returns a pointer to the WorldSpawn instance that owns us. */
	inline WorldSpawn* getWorldSpawn() { return worldSpawn; }

	/* Returns how many chunks were loaded from disk in this frame. */
	inline int getRenderedChunksInThisFrame() { return renderedChunksInThisFrame; }
	inline void incrementRenderedChunksInThisFrame() { ++renderedChunksInThisFrame; }
	inline void resetRenderedChunksInThisFrame() { renderedChunksInThisFrame = 0; }
	inline bool canRenderAnotherChunkInThisFrame() { return renderedChunksInThisFrame < maxRenderedChunksPerFrame; }

	/* Returns the maximum about of chunks that are allowed to be loaded from disk perframe. This is stop
	stuttering that occurs from having too many chunks loading at once.*/
	inline int getMaxRenderedChunksPerFrame() { return maxRenderedChunksPerFrame; }

	/* Converts a chunkId to a ofVec2f containing the x, y position of the chunkId. 
	
	Example:
	World is 4x5
	. . . .
	. . . .
	. . . .
	. . o .
	. . . .

	chunkId of 14 = ofVec2f(2, 3)	
	*/
	inline glm::uvec2 convertChunkIdToVec(int id) { return glm::uvec2(id % numChunksX, id / numChunksX); }

	/* Performs the reverse operation of convertChunkIdToVec().
	Example:
	World is 5x4

	0 1 2 3 4
	5 6 7 o .
	. . . . .
	. . . . .

	chunkPos{3, 1} = chunkId of 8
	*/
	inline int convertChunkVecToId(const glm::uvec2& vec) { return (vec.y * numChunksX + vec.x); }

	/* Returns a ofVec2f containing blockWidth and blockHeight. */
	inline glm::uvec2 getBlockDim() { return glm::uvec2(blockWidth, blockHeight); }

	/* Returns the sizeof each Chunk when they are written to memory. As in
	This is the sizeof(<ChunkMetaData>) + sizeof(<BlockData>) * numberOfBlocksInAChunk. */
	size_t getChunkDataSize();
	
	/* Teporarily creates a world and saves it to disk. Currently is only used for testing. Replace
	for function later on with a proper World Generation algorithm. This is currently called in the
	constructor but obviously when you open a world you don't want this function to be called. */
	void temporaryCreateWorld();

	/* Checks the loadedChunks map cache to see if extra Chunks need to be added or some existing
	ones need to be removed. Currently not implemented correctly. */
	void updateChunks();

	/* Frees a chunk from memory. Does not update the loadedChunks map. */
	void freeChunk(Chunk* chunk);

	/* Free a chunk from memory that is at the chunkPos. Does not update the loadedChunks map. */
	void freeChunk(const glm::uvec2& chunkPos);

	/* Loads a chunk from disk into the loadedChunks cache at the chunkPos specified. Returns the chunk
	at the chunkPos if it is already in the cache. */
	Chunk* loadChunk(const glm::uvec2& chunkPos);

	/* Returns the Block at the worldPos. If the block is not in the loadedChunk cache it is added 
	and then the Block is returned. Currently does not check if the block you are asking for is out of
	bounds and will likely throw an error when you go to read outside the bounds of the world. */
	Block* getBlock(const glm::vec2& worldPos);

	/* Returns a block given a chunkPos and the relative position of the block from the top left of the
	chunk. */
	Block* getBlock(const glm::vec2& chunkPos, const glm::vec2& chunkRelativePos);

	/* Returns the chunk at the chunkPos. If the Chunk is not in the loadedChunk cache it is added and
	then returned. */
	Chunk* getChunk(const glm::vec2& chunkPos);
};


#endif /* WORLD_DATA_H */