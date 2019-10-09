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
	drawWorld();
	popCamera();

	drawOverlay();
}

void WorldSpawn::drawBackground()
{
	// TODO
}

void WorldSpawn::drawWorld()
{
	getEntityController()->draw();
}

void WorldSpawn::drawOverlay()
{
	// TODO
}

void WorldSpawn::pushCamera()
{
	getEntityController()->getPlayer()->getCamera()->pushCameraMatrix();
}

void WorldSpawn::popCamera()
{
	getEntityController()->getPlayer()->getCamera()->popCameraMatrix();
}
