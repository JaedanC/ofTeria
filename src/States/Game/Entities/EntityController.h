#pragma once
#ifndef ENTITY_CONTROLLER_H
#define ENTITY_CONTROLLER_H

#include "ofMain.h"

class WorldSpawn;
class Player;
class EntityController {
private:
	weak_ptr<WorldSpawn> worldSpawn;
	shared_ptr<Player> player;

public:
	EntityController(weak_ptr<WorldSpawn> worldSpawn);

	inline weak_ptr<WorldSpawn> getWorldSpawn();
	inline weak_ptr<Player> getPlayer();

	void update();
	void draw();
};

#endif /* ENTITY_CONTROLLER_H */
