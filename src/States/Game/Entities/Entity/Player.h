#pragma once
#ifndef PLAYER_H
#define PLAYER_H

#include "ofMain.h"
#include "Entity.h"
#include "Camera/Camera.h"

class EntityController;
class Player : public Entity, public enable_shared_from_this<Player> {
private:
	/* Controls the camera within the world. WorldData reads this often. */
	shared_ptr<Camera> camera;

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
	virtual void draw();
};

#endif /* PLAYER_H */