#pragma once
#ifndef WORLD_SPAWN_H
#define WORLD_SPAWN_H
#include "ofMain.h"
#include "WorldData/WorldData.h"
#include "Entities/EntityController.h"

class WorldSpawn {
private:
	string worldName;
	WorldData worldData;
	EntityController entityController;

public:
	WorldSpawn(const string& worldName);
	ofVec2f const convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom=1);

	inline string& getWorldName() { return worldName; }
	inline WorldData* getWorldData() { return &worldData; }
	inline EntityController* getEntityController() { return &entityController; }

	void setup(const string& newWorldName);
	void update();
	void draw();

private:
	void drawBackground();
	void drawForeground();
	void drawOverlay();

	void pushCamera();
	void popCamera();
};

#endif /* WORLD_SPAWN_H */