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
	Camera(weak_ptr<Player> player);
	inline weak_ptr<Player> getPlayer();
	inline ofVec2f* getWorldPos();
	inline void setZoom(float zoom);

	void pushCameraMatrix();
	void popCameraMatrix();
};


#endif /* CAMERA_H */