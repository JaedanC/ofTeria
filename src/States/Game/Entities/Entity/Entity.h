#pragma once
#ifndef ENTITY_H
#define ENTITY_H

#include "ofMain.h"
#include "Hitbox.h"

class EntityController;
class Entity {
private:
	virtual void update() = 0;

protected:
	/* Pointer to the EntityController instance that owns us. */
	EntityController* entityController;

	/* Generic worldPos of any entity instance. */
	ofVec2f worldPos;

	/* Hitboxes. */
	Hitbox hitbox;

	ofVec2f velocity;

public:
	/* Takes in a pointer to the EntityController instance that owns us. */
	Entity(EntityController* entityController);

	/* Virtual destructor allows safe destroying of classed that inherit from this base class. */
	virtual ~Entity() {
		cout << "Destroying Entity\n";
	}

	/**/
	inline Hitbox* getHitbox() { return &hitbox; }

	/**/
	inline ofVec2f* getVelocity() { return &velocity; }

	/* Returns the worldPos of this Entity. */
	inline ofVec2f* getWorldPos() { return &worldPos; }

	/* Returns a Pointer to the EntityController instance that owns us. */
	inline EntityController* getEntityController() { return entityController; }

	/* All classes that inherit from this base class need to implement these. See
	update() which is private. */
	void updateEntity();
	virtual void draw() = 0;
	
	// TODO: SpriteController
	// TODO: vector<Hitbox> hitboxes
};

#endif /* ENTITY_H */