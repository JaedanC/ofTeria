#include "WorldData.h"

WorldData::WorldData(const string& worldName)
{
}

inline ofVec2f WorldData::convertChunkIdToVec(int id)
{
	return ofVec2f();
}

inline int WorldData::convertChunkVecToId(const ofVec2f& vec)
{
	return 0;
}

inline size_t WorldData::getChunkDataSize()
{
	return size_t();
}

inline ofVec2f WorldData::getBlockDim()
{
	return ofVec2f();
}

void WorldData::updateChunks()
{
}
