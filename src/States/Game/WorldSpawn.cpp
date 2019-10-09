#include "WorldSpawn.h"

WorldSpawn::WorldSpawn(const string& worldName)
	: worldName(worldName), worldData(worldName, this), entityController(this)
{

}

ofVec2f const WorldSpawn::convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom)
{
	return cameraWorldPos + screenPos / zoom;
}

void WorldSpawn::setup(const string& newWorldName)
{
}

void WorldSpawn::update()
{
	getWorldData()->updateChunks();
	getEntityController()->update();
}

void WorldSpawn::draw()
{
	drawBackground();

	pushCamera();
	drawForeground();
	popCamera();

	drawOverlay();
}

void WorldSpawn::drawBackground()
{
}

void WorldSpawn::drawForeground()
{
	getEntityController()->draw();
}

void WorldSpawn::drawOverlay()
{
}

void WorldSpawn::pushCamera()
{
}

void WorldSpawn::popCamera()
{
}
