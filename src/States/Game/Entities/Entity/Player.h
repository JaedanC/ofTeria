#pragma once
#ifndef PLAYER_H
#define PLAYER_H

#include "ofMain.h"
#include "Entity.h"

class Camera;
class EntityController;
class Player : public Entity {
private:
	shared_ptr<Camera> camera;

public:
	Player(weak_ptr<EntityController> entityController);

	inline weak_ptr<Camera> getCamera();
	virtual void update();
	virtual void draw();
};

#endif /* PLAYER_H */