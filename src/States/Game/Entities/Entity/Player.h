#pragma once
#ifndef PLAYER_H
#define PLAYER_H

#include "ofMain.h"
#include "Entity.h"
#include "Camera/Camera.h"

class EntityController;
class Player : public Entity, public enable_shared_from_this<Player> {
private:
	shared_ptr<Camera> camera;

public:
	Player(EntityController* entityController);

	inline weak_ptr<Camera> getCamera() { return camera; }
	virtual void update();
	virtual void draw();
};

#endif /* PLAYER_H */