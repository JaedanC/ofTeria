#pragma once
#ifndef ENTITY_H
#define ENTITY_H

#include "ofMain.h"

class EntityController;
class Entity {
protected:
	EntityController* entityController;
	ofVec2f worldPos;

public:
	Entity(EntityController* entityController);
	virtual ~Entity() {}

	inline ofVec2f* getWorldPos() { return &worldPos; }
	inline EntityController* getEntityController() { return entityController; }
	// TODO: SpriteController
	// TODO: vector<Hitbox> hitboxes

	virtual void update() = 0;
	virtual void draw() = 0;
	
};

#endif /* ENTITY_H */