#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "ofMain.h"

class Player;
class Camera {
private:
	weak_ptr<Player> player;
	ofVec2f offsetPos;
	float zoom = 1;

public:
	Camera(Player* player);
	inline weak_ptr<Player> getPlayer() { return player; }
	inline void setZoom(float zoom_) { zoom = zoom_; }

	ofVec2f* getPlayerPos();
	void pushCameraMatrix();
	void popCameraMatrix();
};


#endif /* CAMERA_H */