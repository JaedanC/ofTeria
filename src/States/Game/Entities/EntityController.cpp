#include "EntityController.h"
#include "../WorldSpawn.h"
#include "../addons/ofxDebugger/ofxDebugger.h"

EntityController::EntityController(WorldSpawn* worldSpawn)
	: worldSpawn(worldSpawn), player(make_shared<Player>(this))
{
	cout << "Constructing EntityController\n";
}

void EntityController::update()
{
	getPlayer().lock()->updateEntity();
}

void EntityController::fixedUpdate()
{
	getPlayer().lock()->fixedUpdateEntity();
}

void EntityController::draw()
{
	getPlayer().lock()->drawEntity();
}
