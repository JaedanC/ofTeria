#pragma once
#ifndef WORLD_SPAWN_H
#define WORLD_SPAWN_H

#include "ofMain.h"
#include "Entities/EntityController.h"
#include "WorldData/WorldData.h"

/* This class is instanciated inside the PlayState. If worldSpawn is destroyed, it should in theory.
Call all the destructors down the tree. Maybe try and use some more smart pointers to automate this
process. */
class WorldSpawn : public enable_shared_from_this<WorldSpawn> {
private:
	/* Name of the world. */
	string worldName;

	/* Controls the Entities in the game. */
	shared_ptr<EntityController> entityController;

	/* Controls the WorldData. */
	shared_ptr<WorldData> worldData;

public:
	WorldSpawn(const string& worldName);
	~WorldSpawn() {
		cout << "Destroying WorldSpawn\n";
	};

	/* This function takes in a position on the screen and converts it to world coordinates. Things should be
	kept in WorldPos format if they wish to be drawn otherwise you will never see them on the screen. See draw().
	Only things inside drawWorld() are mapped using the player's camera, otherwise, drawBackground() and
	drawOverlay use screenPos. This could be changed later on. */
	ofVec2f const convertScreenPosToWorldPos(ofVec2f& screenPos);

	/* Get the name of the world. */
	inline string& getWorldName() { return worldName; }

	/* Get a weak pointer to the worldData object that I own. */
	inline weak_ptr<WorldData> getWorldData() { return worldData; }

	/* Get a weak pointer to the EntityController object that I own.*/
	inline weak_ptr<EntityController> getEntityController() { return entityController; }

	/* Maybe use this function to initialise a world? Unused at the moment. It current just changes the name
	of the world. */
	void setup(const string& newWorldName);

	/* Updates all my members that have an update() function. */
	void update();
	void fixedUpdate();

	/* Calls the drawBackground(), drawWorld(), drawOverlay() functions. drawWorld uses the Player's camera so you can
	draw straight to worldPos. */
	void draw();

private:
	void drawBackground();
	void drawWorld();
	void drawOverlay();

	/* Call the Player's camera pushCameraMatrix() so that we can draw straight to worldPos and let the camera
	transformation move it to the screen. */
	void pushCamera();

	/* Returns drawing back to usual. */
	void popCamera();
};

#endif /* WORLD_SPAWN_H */