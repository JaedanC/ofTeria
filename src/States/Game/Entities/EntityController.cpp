#include "EntityController.h"

EntityController::EntityController(WorldSpawn* worldSpawn)
	: worldSpawn(worldSpawn), player(this)
{

}

inline WorldSpawn* EntityController::getWorldSpawn()
{
	return worldSpawn;
}

inline Player* EntityController::getPlayer()
{
	return &player;
}

void EntityController::update()
{
	getPlayer()->update();
}

void EntityController::draw()
{
	getPlayer()->draw();
}
