#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "ofMain.h"

class Player;
class Camera {
private:
	Player* player;
	ofVec2f offsetPos;
	float zoom = 1;

public:
	Camera(Player* player);
	~Camera() {
		cout << "Destroying Camera\n";
	}

	inline Player* getPlayer() { return player; }
	inline void setZoom(float zoom_) { zoom = zoom_; }
	inline float& getZoom() { return zoom; }

	ofVec2f* getPlayerPos();
	void update();
	void pushCameraMatrix();
	void popCameraMatrix();
};


#endif /* CAMERA_H */