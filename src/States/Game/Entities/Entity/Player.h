#pragma once
#ifndef PLAYER_H
#define PLAYER_H

#include "ofMain.h"
#include "Entity.h"
#include "Camera/Camera.h"

class Player : public Entity {
private:
	Camera camera;
public:
	Player();

	inline Camera* getCamera() { return &camera; }
};

#endif /* PLAYER_H */