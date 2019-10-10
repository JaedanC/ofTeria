#include "EntityController.h"
#include "../WorldSpawn.h"

EntityController::EntityController(WorldSpawn* worldSpawn)
	: worldSpawn(worldSpawn->weak_from_this()), player(make_shared<Player>(this))
{
	cout << "Making EntityController\n";
}

void EntityController::update()
{
}

void EntityController::draw()
{
}
