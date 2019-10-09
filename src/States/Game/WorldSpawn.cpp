#include "WorldSpawn.h"
#include "WorldData/WorldData.h"
#include "Entities/EntityController.h"
#include "Entities/Entity/Player.h"
#include "Entities/Entity/Camera/Camera.h"

WorldSpawn::WorldSpawn(const string& worldName)
	: worldName(worldName), worldData(make_shared<WorldData>(worldName, this)), entityController(make_shared<EntityController>(this))
{

}

ofVec2f const WorldSpawn::convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom)
{
	return cameraWorldPos + screenPos / zoom;
}

inline string& WorldSpawn::getWorldName()
{
	return worldName;
}

inline weak_ptr<WorldData> WorldSpawn::getWorldData()
{
	return worldData;
}

inline weak_ptr<EntityController> WorldSpawn::getEntityController()
{
	return entityController;
}

void WorldSpawn::setup(const string& newWorldName)
{
}

void WorldSpawn::update()
{
	getWorldData().lock()->updateChunks();
	getEntityController().lock()->update();
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
	getEntityController().lock()->draw();
}

void WorldSpawn::drawOverlay()
{
	// TODO
}

void WorldSpawn::pushCamera()
{
	getEntityController().lock()->getPlayer().lock()->getCamera().lock()->pushCameraMatrix();
}

void WorldSpawn::popCamera()
{
	getEntityController().lock()->getPlayer().lock()->getCamera().lock()->popCameraMatrix();
}
