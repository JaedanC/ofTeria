#include "WorldSpawn.h"

ofVec2f const WorldSpawn::convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom)
{
	return cameraWorldPos + screenPos / zoom;
}

void WorldSpawn::draw()
{
	drawBackground();

	pushCamera();
	drawForeground();
	popCamera();

	drawOverlay();
}
