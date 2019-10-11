#pragma once
#ifndef ENTITY_CONTROLLER_H
#define ENTITY_CONTROLLER_H

#include "ofMain.h"
#include "Entity/Player.h"

class WorldSpawn;
class EntityController : public enable_shared_from_this<EntityController> {
private:
	/* Pointer to the WorldSpawn instance that owns us. */
	WorldSpawn* worldSpawn;

	/* The player inside the game. */
	shared_ptr<Player> player;

public:
	/* Takes in a pointer to the WorldSpawn instance that owns us. */
	EntityController(WorldSpawn* worldSpawn);
	~EntityController() {
		cout << "Destroying EntityController\n";
	}

	/* Returns a pointer to the WorldSpawn instance that owns us. */
	inline WorldSpawn* getWorldSpawn() { return worldSpawn; }

	/* Returns a weakptr to the Player instance that we own. */
	inline weak_ptr<Player> getPlayer() { return player; }

	/* Calls all the update/draw functions of the Entities in the world. Currently only the player
	exists but later this will be a vector<Entity>. */
	void update();
	void draw();
};

#endif /* ENTITY_CONTROLLER_H */
