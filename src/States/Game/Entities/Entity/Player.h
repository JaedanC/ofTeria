#pragma once
#ifndef PLAYER_H
#define PLAYER_H

#include "ofMain.h"
#include "Entity.h"
#include "Camera/Camera.h"

class EntityController;
/* The player class stores all the data about a single player in the World.
Stores a camera which is used for many calculations. If this game was to become
multiplayer, it would need to make all references through getEntityController()
use an intermediary function to get the specific player you want. Remember,
many useful global function that all Entities could use should go in the
Entity class not in here. */
class Player : public Entity, public enable_shared_from_this<Player> {
private:
	/* Controls the camera within the world. WorldData reads this often. */
	shared_ptr<Camera> camera;

	float width = 25;
	float height = 37;
public:
	/* Takes in a pointer to the EntityController instance that owns us. */
	Player(EntityController* entityController);
	~Player() {
		cout << "Destroying Player\n";
	}

	/* Returns a weakptr to the Camera instance that we own. */
	inline weak_ptr<Camera> getCamera() { return camera; }

	/* Inherited abstract functions from the Entity Class. */
	virtual void update();
	virtual void fixedUpdate();
	virtual void draw();
};

#endif /* PLAYER_H */