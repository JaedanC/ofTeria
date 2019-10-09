#pragma once
#ifndef PLAYER_H
#define PLAYER_H

#include "ofMain.h"
#include "Entity.h"
#include "Camera/Camera.h"

class EntityController;
class Player : public Entity {
private:
	Camera camera;

public:
	Player(EntityController* entityController);

	inline Camera* getCamera();
	inline EntityController* getEntityController();

	virtual void update();
	virtual void draw();
};

#endif /* PLAYER_H */