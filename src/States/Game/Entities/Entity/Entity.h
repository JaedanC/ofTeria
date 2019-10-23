#pragma once
#ifndef ENTITY_H
#define ENTITY_H

#include "ofMain.h"
#include "Hitbox.h"

class EntityController;
/* The Entity class is the base class for all Entities in the world. Movement is
automically handled here and it is done via the fixedUpdate() function. To change
the velocity use getVelocity and dereference the pointer. Interpolation of the Entities
movement is also calculated. To draw at the correct location for this Entity use
getInterp(). This will smooth out the input. You do not need to worry about the Player's
interpolation when drawing as this is handled by the Camera class. */
class Entity {
protected:
	/* All entities must define classes. Note: generic drawing of the Hitbox and Sprite
	are already handled in the Master draw() for Entities. */
	virtual void fixedUpdate() = 0;
	virtual void update() = 0;
	virtual void draw() = 0;

	void calculateCollision();

	/* Pointer to the EntityController instance that owns us. */
	EntityController* entityController;

	/* Generic worldPos of any entity instance. */
	ofVec2f worldPos;
	ofVec2f prevWorldPos;
	ofVec2f interp;

	/* Hitboxes. */
	Hitbox hitbox;

	/* Vector that stores the velocity of the Entity*/
	ofVec2f velocity;

	/* Entity Flags */
	bool affectedByGravity;

public:
	/* Takes in a pointer to the EntityController instance that owns us. */
	Entity(EntityController* entityController, bool affectedByGravity);

	/* Virtual destructor allows safe destroying of classed that inherit from this base class. */
	virtual ~Entity() {
		cout << "Destroying Entity\n";
	}

	/* Returns a pointer to the Entities hitbox. 
	TODO: Change this to be a list of Hitboxes. */
	inline Hitbox* getHitbox() { return &hitbox; }

	/* Returns a pointer to the Entities Velocity vector*/
	inline ofVec2f* getVelocity() { return &velocity; }

	/* Adding a velocity. */
	inline void addVelocity(ofVec2f& v) { velocity += v; }

	/* Returns the previousWorldPos of this Entity. This is usually the
	function you want to use not getWorldPos. */
	inline ofVec2f* getPrevWorldPos() { return &prevWorldPos; }
	inline ofVec2f* getWorldPos() { return &worldPos; }

	/* Returns a vector to the offset required to correctly draw the Entity without
	stuttering on the screen. Example usage:
		ofDrawRectangle(<some-worldPos> + *getInterp(), w, h);
	*/
	inline ofVec2f* getInterp() { return &interp; }

	/* Returns a Pointer to the EntityController instance that owns us. */
	inline EntityController* getEntityController() { return entityController; }

	/* When updating in entity altogether use these functions as they will do some calculation
	for the Entity as a whole automically and then call the corresponding fixedUpdate(), update()
	and draw(), (which is protected) for the child class. FixedUpdate() updates every (1 / tickrate)
	seconds and should be used for physics calculations. */
	void fixedUpdateEntity();
	void updateEntity();
	void drawEntity();
	
	// TODO: SpriteController
	// TODO: vector<Hitbox> hitboxes
};

#endif /* ENTITY_H */