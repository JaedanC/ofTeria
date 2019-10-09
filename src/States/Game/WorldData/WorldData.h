#pragma once
#ifndef WORLD_DATA_H
#define WORLD_DATA_H
#include "ofMain.h"
#include "Chunk.h"

class WorldData {
private:
	vector<Chunk*> loadedChunks;

public:
	WorldData() {}
};


#endif /* WORLD_DATA_H */