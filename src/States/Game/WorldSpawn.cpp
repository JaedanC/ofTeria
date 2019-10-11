#include "WorldSpawn.h"

#include "Entities/EntityController.h"
#include "WorldData/WorldData.h"
#include "Entities/Entity/Player.h"
#include "Entities/Entity/Camera/Camera.h"
#include "../addons/ofxDebugger/ofxDebugger.h"

WorldSpawn::WorldSpawn(const string& worldName)
	: worldData(make_shared<WorldData>(this)), entityController(make_shared<EntityController>(this))
{
	cout << "Constructing WorldSpawn\n";
}

ofVec2f const WorldSpawn::convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom)
{
	ofVec2f worldPos = cameraWorldPos + screenPos / zoom;
	return worldPos;
}

void WorldSpawn::setup(const string& newWorldName)
{
	worldName = newWorldName;
}

void WorldSpawn::update()
{
	getEntityController().lock()->update();
	getWorldData().lock()->updateChunks();
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
}

void WorldSpawn::drawWorld()
{
	worldData->draw();
	entityController->draw();
}

void WorldSpawn::drawOverlay()
{
	ofSetColor(ofColor::yellow);
	ofDrawRectangle(500, 50, 50, 50);
}

void WorldSpawn::pushCamera()
{
	getEntityController().lock()->getPlayer().lock()->getCamera().lock()->pushCameraMatrix();
}

void WorldSpawn::popCamera()
{
	getEntityController().lock()->getPlayer().lock()->getCamera().lock()->popCameraMatrix();
}
