#pragma once
#ifndef WORLD_SPAWN_H
#define WORLD_SPAWN_H

#include "ofMain.h"
#include "Entities/EntityController.h"
#include "WorldData/WorldData.h"

class WorldSpawn : public enable_shared_from_this<WorldSpawn> {
private:
	string worldName;
	shared_ptr<EntityController> entityController;
	shared_ptr<WorldData> worldData;

public:
	WorldSpawn(const string& worldName);

	ofVec2f const convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom=1);
	inline string& getWorldName() { return worldName; }
	inline weak_ptr<WorldData> getWorldData() { return worldData; }
	inline weak_ptr<EntityController> getEntityController() { return entityController; }

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