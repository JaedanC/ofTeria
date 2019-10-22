#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "ofMain.h"

class Player;
class Camera {
private:
	Player* player;

	/* This variable stores the offset of the camera relative to the player. By default this will be half the screen
	width and half the screen height however this could easily be changed to support independant movement of the camera
	if the player has a sniper rifle (from terraria) or gets close to the edge of the world. */
	ofVec2f offsetPos;

	/* This is the zoom factor. High numbers indicate more zoomed out. So technically this should be called 'scale'. */
	float zoom = 1;

public:
	/* Takes in a pointer to the Player instance that owns us. */
	Camera(Player* player);
	~Camera() {
		cout << "Destroying Camera\n";
	}

	/* Returns a pointer to the Player instance that owns us. */
	inline Player* getPlayer() { return player; }

	/* Changes the zoom factor of the Camera. */
	inline void setZoom(float zoom_) {
		zoom = ofClamp(zoom_, 0.2, 5);
	}

	/* Returns the zoom factor of the Camera. */
	inline float getZoom() { return zoom; }

	/* Returns a ofVec2f* pointing to the players exact position in the world. */
	ofVec2f* getPlayerPos();

	/* Before drawing anything using it's worldPositon you should first call this function. This maps the world coordinate
	relative to the camera while taking into consideration the zoom factor. */
	void pushCameraMatrix();

	/* Once you are finshed drawing worldPosition's you should call this function. */
	void popCameraMatrix();
};

#endif /* CAMERA_H */