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
	/* Room for optimisation? 
	Release @ 256 loaded chunks getting roughly 65fps

	int blockWidth = 32;
	int blockHeight = 32;

	int chunkWidth = 16;
	int chunkHeight = 16;

	int worldWidth = 1024;
	int worldHeight = 1024;

	int screenChunkLoadWidth = 500;
	int screenChunkLoadHeight = 500;
	*/
	//ofNoFill();
	ofDisableAlphaBlending();
	for (auto& pair : loadedChunks) {
		Chunk* chunk = pair.second;
		ofVec2f& chunkPos = chunk->getChunkMetaData()->chunkPos;
		int chunkOffsetX = chunkPos.x * chunkWidth * blockWidth;
		int chunkOffsetY = chunkPos.y * chunkHeight * blockHeight;


		int chunkWidth = chunk->getChunkMetaData()->chunkWidth;
		int chunkHeight = chunk->getChunkMetaData()->chunkHeight;
		int x, y;

		for (int i = 0; i < chunk->getChunkMetaData()->numBlocks; i++) {
			x = blockWidth * (i % chunkWidth) + chunkOffsetX;
			y = blockHeight * (i / chunkHeight) + chunkOffsetY;
			ofSetColor(abs(200 - (x / blockWidth) / 2 - (y / blockHeight) / 2) % 255 + 1, ((x / blockWidth) * 4 % 255), ((y / blockHeight) * 4 % 255));
			ofDrawRectangle(x, y, blockWidth, blockHeight);
		}

		/*c.setHsb((int)((chunkPos.x + chunkPos.y) * 100 + 200) % 255, 255, 255, 20);
		ofSetColor(c);
		ofDrawRectangle(chunkOffsetX, chunkOffsetY, chunkWidth * blockWidth, chunkHeight * blockHeight);*/
	}

	/* Draw the chunk loading rectangle. */
	//ofVec2f* playerPos = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getWorldPos();
	//float& zoom = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getCamera().lock()->getZoom();
	//int chunkPixelWidth = chunkWidth * blockWidth;
	//int chunkPixelHeight = chunkHeight * blockHeight;
	//int playerLeftChunkBorder	= ((playerPos->x) - (screenChunkLoadWidth / 2) * zoom);
	//int playerRightChunkBorder	= ((playerPos->x) + (screenChunkLoadWidth / 2) * zoom);
	//int playerUpChunkBorder		= ((playerPos->y) - (screenChunkLoadHeight / 2) * zoom);
	//int playerDownChunkBorder	= ((playerPos->y) + (screenChunkLoadHeight / 2) * zoom);
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
	//num_chunks, numChunksX, numChunksY
	//chunkWidth, chunkHeight
	int dataSize = getChunkDataSize() * numChunks;
	getWorldFile().lock()->resize(dataSize);

	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		ofVec2f chunkPos = convertChunkIdToVec(chunkId);
		Chunk chunk(chunkPos, chunkWidth, chunkHeight, this);
		chunk.createRandomData();
		chunk.saveChunk();
		chunk.freeData();
	}
}

void WorldData::updateChunks()
{
	{
		ofxTimer timer("updateChunks()");

		// TODO: Implement this function
		ofVec2f* playerPos = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getWorldPos();
		float& zoom = getWorldSpawn()->getEntityController().lock()->getPlayer().lock()->getCamera().lock()->getZoom();

		int chunkPixelWidth = chunkWidth * blockWidth;
		int chunkPixelHeight = chunkHeight * blockHeight;

		int playerLeftChunkBorder = ((playerPos->x) - (screenChunkLoadWidth / 2) * zoom) / chunkPixelWidth;
		int playerRightChunkBorder = ((playerPos->x) + (screenChunkLoadWidth / 2) * zoom) / chunkPixelWidth;
		int playerUpChunkBorder = ((playerPos->y) - (screenChunkLoadHeight / 2) * zoom) / chunkPixelHeight;
		int playerDownChunkBorder = ((playerPos->y) + (screenChunkLoadHeight / 2) * zoom) / chunkPixelHeight;

		// Free the ones outside the range
		vector<int> toDelete;
		for (auto& pair : loadedChunks) {
			ofVec2f chunkPos = convertChunkIdToVec(pair.first);
			if (chunkPos.x < playerLeftChunkBorder || chunkPos.x > playerRightChunkBorder + 1 ||
				chunkPos.y < playerUpChunkBorder || chunkPos.y > playerDownChunkBorder + 1)
			{
				toDelete.push_back(pair.first);
			}
		}
		for (int chunkId : toDelete) {
			freeChunk(loadedChunks[chunkId]);
			loadedChunks.erase(chunkId);
		}

		// Get the correct chunks
		for (int i = playerLeftChunkBorder; i < playerRightChunkBorder + 1; i++) {
			for (int j = playerUpChunkBorder; j < playerDownChunkBorder + 1; j++) {
				loadChunk({ (float)i, (float)j });
			}
		}

		debugPush("LoadedChunks: " + ofToString(loadedChunks.size()));

	}
}

void WorldData::freeChunk(Chunk* chunk)
{
	ofVec2f& chunkPos = chunk->getChunkMetaData()->chunkPos;
	freeChunk(chunkPos);
}

void WorldData::freeChunk(const ofVec2f& chunkPos)
{
	int chunkId = convertChunkVecToId(chunkPos);
	if (loadedChunks.count(chunkId) == 0) {
		cout << "WorldData::freeChunk: chunkDoes not exist. Can't free\n";
		return;
	}

	Chunk* chunk = loadedChunks[chunkId];
	chunk->freeData();
	delete[] chunk;
}

Chunk* WorldData::loadChunk(const ofVec2f& chunkPos)
{
	if (chunkPos.x >= numChunksX || chunkPos.y >= numChunksY ||
		chunkPos.x < 0 || chunkPos.y < 0) {
		return nullptr;
	}
	int chunkId = convertChunkVecToId(chunkPos);

	if (loadedChunks.count(chunkId) != 0) {
		//cout << "WorldData::loadChunk: Chunk already loaded at " << chunkPos << endl;
		return loadedChunks[chunkId];
	}

	int dataSize = getChunkDataSize();
	int offset = chunkId * dataSize;

	Chunk* chunk = new Chunk(chunkPos, chunkWidth, chunkHeight, this);
	chunk->loadChunk(chunkId);

	loadedChunks[chunkId] = chunk;
	return chunk;
}

Block* WorldData::getBlock(const ofVec2f& worldPos)
{
	ofVec2f chunkPos = {
		worldPos.x / chunkWidth,
		worldPos.y / chunkHeight
	};

	int a = static_cast<int>(worldPos.x) % chunkWidth;
	int b = static_cast<int>(worldPos.y) % chunkHeight;

	ofVec2f chunkRelativePos = { (float)a, (float)b };

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
	int chunkId = convertChunkVecToId(chunkPos);

	if (loadedChunks.count(chunkId) != 0) {
		chunk = loadedChunks[chunkId];
		return chunk;
	}

	return loadChunk(chunkPos);
}
