#pragma once
#ifndef WORLD_SPAWN_H
#define WORLD_SPAWN_H
#include "ofMain.h"

class WorldSpawn {
private:
	string worldName;

public:
	WorldSpawn(const string& worldName)
		: worldName(worldName)
	{
		
	}

	ofVec2f const convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom=1);

	inline string& getWorldName() { return worldName; }

	void setup(const string& newWorldName);
	void update();
	void draw();

private:
	void drawWorld();
	void drawScreen();
	void pushCamera();
	void popCamera
};

#endif /* WORLD_SPAWN_H */