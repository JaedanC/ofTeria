#include "WorldData.h"
#include "ofMain.h"
#include "ofVec2f.h"
#include "Chunk.h"
#include "../WorldSpawn.h"
#include "../Entities/EntityController.h"
#include "../Entities/Entity/Entity.h"
#include "../Entities/Entity/Player.h"
#include "../Entities/Entity/Camera/Camera.h"
#include "ofxTimer/ofxTimer.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ofxMemoryMapping/ofxMemoryMapping.h"
#include "ofxTimer/ofxTimer.h"


WorldData::WorldData(WorldSpawn* worldSpawn)
	: worldSpawn(worldSpawn), worldFile(make_shared<ofxMemoryMapping>(worldSpawn->getWorldName()))
{
	cout << "Constructing WorldData\n";

	numChunksX = ceil((float)worldWidth / chunkWidth);
	numChunksY = ceil((float)worldHeight / chunkHeight);
	numChunks = numChunksX * numChunksY;

	temporaryCreateWorld();
}

void WorldData::draw()
{
	ofxTimer timer("WorldData::draw()");

	for (auto& pair : loadedChunks) {
		Chunk* chunk = pair.second;
		glm::uvec2& chunkPos = chunk->getChunkMetaData()->chunkPos;
		int chunkOffsetX = chunkPos.x * chunkWidth * blockWidth;
		int chunkOffsetY = chunkPos.y * chunkHeight * blockHeight;

		chunk->drawChunk(chunkOffsetX, chunkOffsetY);
	}

	/* Draw the chunk loading rectangle. */
	//ofVec2f* playerPos = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getWorldPos();
	//float& zoom = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getCamera().lock()->getZoom();
	//int chunkPixelWidth = chunkWidth * blockWidth;
	//int chunkPixelHeight = chunkHeight * blockHeight;
	//int playerLeftChunkBorder	= ((playerPos->x) - (ofGetWidth() / 2) * zoom);
	//int playerRightChunkBorder	= ((playerPos->x) + (ofGetWidth() / 2) * zoom);
	//int playerUpChunkBorder		= ((playerPos->y) - (ofGetHeight() / 2) * zoom);
	//int playerDownChunkBorder	= ((playerPos->y) + (ofGetHeight() / 2) * zoom);
	//
	//ofSetColor(ofColor::black, 255);
	//ofNoFill();
	//ofDrawRectangle(playerLeftChunkBorder, playerUpChunkBorder, playerRightChunkBorder - playerLeftChunkBorder, playerDownChunkBorder - playerUpChunkBorder);
	//ofFill();
}

size_t WorldData::getChunkDataSize()
{
	return sizeof(ChunkSaved) + chunkWidth * chunkHeight * sizeof(Block);
}

void WorldData::temporaryCreateWorld()
{
	cout << "WorldData::temporaryCreateWorld: Creating world\n";
	//num_chunks, numChunksX, numChunksY
	//chunkWidth, chunkHeight
	int dataSize = getChunkDataSize() * numChunks;
	getWorldFile().lock()->resize(dataSize);

	for (unsigned int chunkId = 0; chunkId < numChunks; chunkId++) {
		glm::uvec2 chunkPos = convertChunkIdToVec(chunkId);
		Chunk chunk(chunkPos, chunkWidth, chunkHeight, this);
		chunk.createRandomData();
		chunk.saveChunk();
	}
}

void WorldData::updateChunks()
{
	{
		ofxTimer timer("updateChunks()");
		resetRenderedChunksInThisFrame();

		ofVec2f* playerPos = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getPrevWorldPos();
		float zoom = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getCamera().lock()->getZoom();

		/* Calculate where the chunk loading borders are from the player. */
		int chunkPixelWidth = chunkWidth * blockWidth;
		int chunkPixelHeight = chunkHeight * blockHeight;
		int playerLeftChunkBorder =		((playerPos->x) - (ofGetWidth() / 2) * zoom) / chunkPixelWidth;
		int playerRightChunkBorder =	((playerPos->x) + (ofGetWidth() / 2) * zoom) / chunkPixelWidth;
		int playerUpChunkBorder =		((playerPos->y) - (ofGetHeight() / 2) * zoom) / chunkPixelHeight;
		int playerDownChunkBorder =		((playerPos->y) + (ofGetHeight() / 2) * zoom) / chunkPixelHeight;

		/* Add the chunkId's to need to be deleted to a list to avoid editing the loadedChunks container
		while iterating through it. */
		vector<int> toDelete;
		for (auto& pair : loadedChunks) {
			ofVec2f chunkPos = convertChunkIdToVec(pair.first);
			if (chunkPos.x < playerLeftChunkBorder || chunkPos.x > playerRightChunkBorder + 1 ||
				chunkPos.y < playerUpChunkBorder || chunkPos.y > playerDownChunkBorder + 1)
			{
				toDelete.push_back(pair.first);
			}
		}

		/* Remove these chunks from memory. */
		for (int chunkId : toDelete) {
			freeChunk(loadedChunks[chunkId]);
			loadedChunks.erase(chunkId);
		}

		/* Loads the chunks from memory that are supposed to be on the screen */
		for (int i = playerLeftChunkBorder; i < playerRightChunkBorder + 1; i++) {
			for (int j = playerUpChunkBorder; j < playerDownChunkBorder + 1; j++) {
				loadChunk({ (float)i, (float)j });
			}
		}

		debugPush("LoadedChunks: " + ofToString(loadedChunks.size()));
		debugPush("LoadedBlocks: " + ofToString(loadedChunks.size() * chunkWidth * chunkHeight));
	}
}

void WorldData::freeChunk(Chunk* chunk)
{
	freeChunk(chunk->getChunkMetaData()->chunkPos);
}

void WorldData::freeChunk(const glm::uvec2& chunkPos)
{
	int chunkId = convertChunkVecToId(chunkPos);
	/* Check to see if this chunk is not loaded. */
	if (loadedChunks.count(chunkId) == 0) {
		cout << "WorldData::freeChunk: chunkDoes not exist. Can't free\n";
		return;
	}

	/* Unload the chunk. */
	delete loadedChunks[chunkId];
}

Chunk* WorldData::loadChunk(const glm::uvec2& chunkPos)
{
	/* Ignore requests to load chunks outside the game world. */
	if (chunkPos.x >= numChunksX || chunkPos.y >= numChunksY ||
		chunkPos.x < 0 || chunkPos.y < 0) {
		return nullptr;
	}
	
	int chunkId = convertChunkVecToId(chunkPos);

	/* Only load the chunk from memory if it is not stored in the cache. If it's in the cache just return
	that instead. */
	if (loadedChunks.count(chunkId) != 0) {
		return loadedChunks[chunkId];
	}

	/* Limit the amount of chunks we can load from memory every frame to stop stuttering as you move through
	the world. On a slow pc this process could be visible if maxRenderPerFrame is too low. Having a lower value
	gives you a steadier fps but a slow pc would not be able to render fast enough to make good usage of this.
	So far maxRenderPerFrame is a small number, and instead we have semi-larger chunks. */
	if (!canRenderAnotherChunkInThisFrame()) {
		return nullptr;
	}
	incrementRenderedChunksInThisFrame();

	/* Load the chunk from memory and allocate enough space for the chunk on the heap. */
	int offset = chunkId * getChunkDataSize();
	Chunk* chunk = new Chunk(chunkPos, chunkWidth, chunkHeight, this);
	loadedChunks[chunkId] = chunk;
	return chunk;
}

Block* WorldData::getBlock(const glm::vec2& worldPos)
{
	glm::vec2 chunkPos = {
		floor(worldPos.x / (chunkWidth * blockWidth)),
		floor(worldPos.y / (chunkHeight * blockHeight))
	};

	int chunkRelativeX = floor(worldPos.x);
	int chunkRelativeY = floor(worldPos.y);
	chunkRelativeX %= chunkWidth;
	chunkRelativeY %= chunkHeight;

	glm::vec2 chunkRelativePos = { chunkRelativeX, chunkRelativeY };

	return getBlock(chunkPos, chunkRelativePos);
}

Block* WorldData::getBlock(const glm::vec2& chunkPos, const glm::vec2& chunkRelativePos)
{
	Chunk* chunk = getChunk(chunkPos);
	if (chunk) {
		return chunk->getBlock(chunkRelativePos);
	} else {
		return nullptr;
	}
}

Chunk* WorldData::getChunk(const glm::vec2& chunkPos)
{
	Chunk* chunk;
	int chunkId = convertChunkVecToId(chunkPos);

	if (loadedChunks.count(chunkId) != 0) {
		chunk = loadedChunks[chunkId];
		return chunk;
	}

	return loadChunk(chunkPos);
}
