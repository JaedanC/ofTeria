#pragma once
#ifndef HITBOX_H
#define HITBOX_H

#include "ofMain.h"

class Entity;
class Hitbox {
private:
	Entity* entity;
	ofVec2f entityOffset;
	ofVec2f* entityLock;
	float w, h;
public:
	void setOffset(Entity* entity, ofVec2f& offset, float w, float h);

	void draw();
};

#endif /* HITBOX_H */