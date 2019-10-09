#pragma once
#ifndef WORLD_SPAWN_H
#define WORLD_SPAWN_H

#include "ofMain.h"

class WorldData;
class EntityController;
class WorldSpawn {
private:
	string worldName;
	shared_ptr<WorldData> worldData;
	shared_ptr<EntityController> entityController;

public:
	WorldSpawn(const string& worldName);
	ofVec2f const convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom=1);

	inline string& getWorldName();
	inline weak_ptr<WorldData> getWorldData();
	inline weak_ptr<EntityController> getEntityController();

	void setup(const string& newWorldName);
	void update();
	void draw();

private:
	void drawBackground();
	void drawWorld();
	void drawOverlay();

	void pushCamera();
	void popCamera();
};

#endif /* WORLD_SPAWN_H */