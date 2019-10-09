#pragma once
#ifndef ENTITY_H
#define ENTITY_H

#include "ofMain.h"

class EntityController;
class Entity {
protected:
	weak_ptr<EntityController> entityController;
	ofVec2f worldPos;

public:
	Entity(weak_ptr<EntityController> entityController);
	virtual ~Entity() {}

	inline ofVec2f* getWorldPos();
	inline weak_ptr<EntityController> getEntityController();
	// TODO: SpriteController
	// TODO: vector<Hitbox> hitboxes

	virtual void update() = 0;
	virtual void draw() = 0;
	
};

#endif /* ENTITY_H */