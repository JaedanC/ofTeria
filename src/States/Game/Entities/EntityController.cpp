#include "EntityController.h"
#include "../WorldSpawn.h"

EntityController::EntityController(WorldSpawn* worldSpawn)
	: worldSpawn(worldSpawn->weak_from_this()), player(make_shared<Player>(this))
{
}

void EntityController::update()
{
	getPlayer().lock()->update();
}

void EntityController::draw()
{
	getPlayer().lock()->draw();
}
