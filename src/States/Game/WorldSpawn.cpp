#include "WorldSpawn.h"

#include "Entities/EntityController.h"
#include "WorldData/WorldData.h"
#include "Entities/Entity/Player.h"
#include "Entities/Entity/Camera/Camera.h"
#include "../addons/ofxDebugger/ofxDebugger.h"

WorldSpawn::WorldSpawn(const string& worldName)
	: worldData(make_shared<WorldData>(this)), entityController(make_shared<EntityController>(this)), worldName(worldName)
{
	cout << "Constructing WorldSpawn\n";
}

ofVec2f const WorldSpawn::convertScreenPosToWorldPos(ofVec2f& screenPos)
{
	
	float zoom = getEntityController().lock()->getPlayer().lock()->getCamera().lock()->getZoom();
	ofVec2f* playerPos = getEntityController().lock()->getPlayer().lock()->getWorldPos();
	ofVec2f worldPos = *playerPos + screenPos * zoom;
	worldPos.x -= zoom * ofGetWidth() / 2.0f;
	worldPos.y -= zoom * ofGetHeight() / 2.0f;
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
