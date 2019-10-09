#pragma once
#ifndef ENTITY_CONTROLLER_H
#define ENTITY_CONTROLLER_H

#include "ofMain.h"
#include "Entity/Player.h"

class WorldSpawn;
class EntityController {
private:
	WorldSpawn* worldSpawn;
	Player player;

public:
	EntityController(WorldSpawn* worldSpawn);

	inline WorldSpawn* getWorldSpawn();
	inline Player* getPlayer();

	void update();
	void draw();
};

#endif /* ENTITY_CONTROLLER_H */
