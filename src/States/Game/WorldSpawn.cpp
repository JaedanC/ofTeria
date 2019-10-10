#include "WorldSpawn.h"

WorldSpawn::WorldSpawn(const string& worldName)
{
}

ofVec2f const WorldSpawn::convertScreenPosToWorldPos(ofVec2f& cameraWorldPos, ofVec2f& screenPos, int zoom)
{
	return { 0, 0 };
}

inline string& WorldSpawn::getWorldName()
{
	// TODO: insert return statement here
}

void WorldSpawn::setup(const string& newWorldName)
{
}

void WorldSpawn::update()
{
}

void WorldSpawn::draw()
{
}

void WorldSpawn::drawBackground()
{
}

void WorldSpawn::drawWorld()
{
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
