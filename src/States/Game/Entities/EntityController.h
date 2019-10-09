#pragma once
#ifndef ENTITY_CONTROLLER_H
#define ENTITY_CONTROLLER_H

#include "ofMain.h"
#include "../WorldSpawn.h"
#include "Entity/Player.h"

class EntityController {
private:
	WorldSpawn* worldSpawn;
	Player player;

public:
	EntityController(WorldSpawn* worldSpawn);

	inline WorldSpawn* getWorldSpawn() { return worldSpawn; }
	inline Player* getPlayer() { return &player; }

	void update();
	void draw();
};

#endif /* ENTITY_CONTROLLER_H */
