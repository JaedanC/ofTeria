#pragma once
#ifndef HITBOX_H
#define HITBOX_H

#include "ofMain.h"

class Entity;
/* This hitbox class simply stores data for an Entities hitbox and will contain
functions to assist in collision detection. */
class Hitbox {
private:
	/* Stores a pointer to our parent entity. That the hitbox will lock onto. */
	Entity* entity;

	/* This vector is the offset of the hitbox relative to the exact x, y position
	of the Entity. Eg: is the player is at {0, 0} I want the hitbox to around around
	that position. 
		entryOffset = {-16, -16}
	This assumes the hitbox is 32*32. */
	ofVec2f entityOffset;

	/* Width, Height of the Hitbox. */
	float w, h;
public:
	/* This function sets the values of the hitbox. Remember to call this function at
	least once before using the Hitbox of it's values will not be defined. */
	void setOffset(Entity* entity, ofVec2f& offset, float w, float h);

	/* Draw's the hitbox to the screen as a wireframe. See this function for an example 
	on how to use getInterp(). */
	void draw();
};

#endif /* HITBOX_H */