#include "EntityController.h"
#include "Entity/Player.h"

EntityController::EntityController(weak_ptr<WorldSpawn> worldSpawn)
	: worldSpawn(worldSpawn), player(make_shared<Player>(this))
{

}

inline weak_ptr<WorldSpawn> EntityController::getWorldSpawn()
{
	return worldSpawn;
}

inline weak_ptr<Player> EntityController::getPlayer()
{
	return player;
}

void EntityController::update()
{
	getPlayer().lock()->update();
}

void EntityController::draw()
{
	getPlayer().lock()->draw();
}
