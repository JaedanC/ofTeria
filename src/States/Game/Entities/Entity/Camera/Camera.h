#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "ofMain.h"
#include "../Player.h"

class Camera {
private:
	Player* player;
	ofVec2f offsetPos;

public:
	Camera(Player* player);
	inline ofVec2f* getWorldPos() { return player->getWorldPos(); }
};

#endif /* CAMERA_H */