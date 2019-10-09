#include "WorldSpawn.h"

ofVec2f convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom) const {
	return cameraWorldPos + screenPos / zoom;
}

void WorldSpawn::draw()
{
	pushCamera();
	drawWorld();
	popCamera();

	drawScreen();
}
