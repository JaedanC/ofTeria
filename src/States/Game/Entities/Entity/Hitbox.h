#pragma once
#ifndef HITBOX_H
#define HITBOX_H

#include "ofMain.h"

class Hitbox {
private:
	ofVec2f entityOffset;
	ofVec2f* entityLock;
	float w, h;
public:
	Hitbox() {}
	void set(ofVec2f* entityLock, ofVec2f& offset, float w, float h);

	void draw();
};

#endif /* HITBOX_H */