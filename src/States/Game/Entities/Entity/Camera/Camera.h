#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "ofMain.h"
//#include "../Player.h"

class Player;
class Camera {
private:
	Player* player;
	ofVec2f offsetPos;
	float zoom = 1;

public:
	Camera(Player* player);
	inline Player* getPlayer();
	inline ofVec2f* getWorldPos();
	inline void setZoom(float zoom);

	void pushCameraMatrix();
	void popCameraMatrix();
};


#endif /* CAMERA_H */