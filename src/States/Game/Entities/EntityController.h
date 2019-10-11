#pragma once
#ifndef ENTITY_CONTROLLER_H
#define ENTITY_CONTROLLER_H

#include "ofMain.h"
#include "Entity/Player.h"

class WorldSpawn;
class EntityController : public enable_shared_from_this<EntityController> {
private:
	WorldSpawn* worldSpawn;
	shared_ptr<Player> player;

public:
	EntityController(WorldSpawn* worldSpawn);

	inline WorldSpawn* getWorldSpawn() { return worldSpawn; }
	inline weak_ptr<Player> getPlayer() { return player; }

	void update();
	void draw();
};

#endif /* ENTITY_CONTROLLER_H */
